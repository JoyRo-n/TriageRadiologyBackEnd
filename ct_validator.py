import io
import struct
import zlib
from typing import Optional, Tuple
import numpy as np

WRONG_IMAGE_MESSAGE = "Wrong image please try again"


def _decode_png_pure_python(data: bytes) -> Optional[np.ndarray]:
    """
    Fallback pure-Python PNG decoder using standard library zlib + struct + numpy,
    used only if TensorFlow, Pillow, and OpenCV are unavailable.
    """
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    try:
        pos = 8
        idat = bytearray()
        w = h = color_type = bit_depth = 0
        while pos < len(data):
            length = struct.unpack(">I", data[pos : pos + 4])[0]
            chunk_type = data[pos + 4 : pos + 8]
            chunk_data = data[pos + 8 : pos + 8 + length]
            if chunk_type == b"IHDR":
                w, h, bit_depth, color_type, _, _, _ = struct.unpack(
                    ">IIBBBBB", chunk_data
                )
            elif chunk_type == b"IDAT":
                idat.extend(chunk_data)
            pos += 12 + length

        if bit_depth != 8 or color_type not in (0, 2, 4, 6):
            return None

        raw = zlib.decompress(bytes(idat))
        bpp = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
        stride = w * bpp
        recon = np.zeros((h, stride), dtype=np.uint8)
        idx = 0
        for y in range(h):
            flt = raw[idx]
            idx += 1
            row = np.frombuffer(raw[idx : idx + stride], dtype=np.uint8).copy()
            idx += stride
            if flt == 1:
                for x in range(bpp, stride):
                    row[x] = (int(row[x]) + int(row[x - bpp])) & 0xFF
            elif flt == 2 and y > 0:
                row = (
                    (row.astype(np.int32) + recon[y - 1].astype(np.int32)) & 0xFF
                ).astype(np.uint8)
            elif flt == 3:
                for x in range(stride):
                    a = int(row[x - bpp]) if x >= bpp else 0
                    b = int(recon[y - 1, x]) if y > 0 else 0
                    row[x] = (int(row[x]) + ((a + b) >> 1)) & 0xFF
            elif flt == 4:
                for x in range(stride):
                    a = int(row[x - bpp]) if x >= bpp else 0
                    b = int(recon[y - 1, x]) if y > 0 else 0
                    c = int(recon[y - 1, x - bpp]) if (y > 0 and x >= bpp) else 0
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                    row[x] = (int(row[x]) + pr) & 0xFF
            recon[y] = row

        img = recon.reshape((h, w, bpp))
        if bpp == 1:
            return np.repeat(img, 3, axis=2)
        elif bpp == 2:
            return np.repeat(img[:, :, :1], 3, axis=2)
        elif bpp == 4:
            return img[:, :, :3]
        return img
    except Exception:
        return None


def decode_image_to_rgb_array(contents: bytes) -> Optional[np.ndarray]:
    """
    Decodes raw image bytes (PNG, JPEG, WEBP, BMP, or DICOM) into an (H, W, 3) uint8 numpy array.
    """
    if not contents or len(contents) < 16:
        return None

    # 1. Try TensorFlow if loaded
    try:
        import tensorflow as tf

        tensor = tf.io.decode_image(contents, channels=3, expand_animations=False)
        return tensor.numpy()
    except Exception:
        pass

    # 2. Try Pillow (PIL)
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception:
        pass

    # 3. Try OpenCV (cv2)
    try:
        import cv2

        arr = np.frombuffer(contents, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass

    # 4. Try pydicom for raw .dcm files
    try:
        import pydicom

        ds = pydicom.dcmread(io.BytesIO(contents))
        px = ds.pixel_array.astype(np.float32)
        px_min, px_max = float(np.min(px)), float(np.max(px))
        if px_max > px_min:
            px_norm = ((px - px_min) / (px_max - px_min) * 255.0).astype(np.uint8)
        else:
            px_norm = np.zeros_like(px, dtype=np.uint8)
        if px_norm.ndim == 2:
            return np.stack([px_norm, px_norm, px_norm], axis=-1)
        return px_norm[:, :, :3]
    except Exception:
        pass

    # 5. Pure-Python PNG fallback
    return _decode_png_pure_python(contents)


def validate_lung_ct_scan(contents: bytes, filename: str = "") -> Tuple[bool, str]:
    """
    Validates whether an uploaded image is a genuine axial Lung / Chest CT scan.
    If the image is NOT a Lung CT scan (e.g. a selfie, colorful photo, document, chart,
    landscape, or non-chest image), returns (False, "Wrong image please try again").
    """
    rgb_uint8 = decode_image_to_rgb_array(contents)
    if rgb_uint8 is None or rgb_uint8.ndim != 3 or rgb_uint8.shape[2] < 3:
        return False, WRONG_IMAGE_MESSAGE

    h, w, _ = rgb_uint8.shape
    if h < 64 or w < 64:
        return False, WRONG_IMAGE_MESSAGE

    aspect_ratio = float(w) / float(h)
    if aspect_ratio < 0.45 or aspect_ratio > 2.4:
        return False, WRONG_IMAGE_MESSAGE

    rgb = rgb_uint8[:, :, :3].astype(np.float32) / 255.0

    # 1. Color / Saturation Check:
    # Lung CT scans are grayscale (or near-grayscale when photographed off a medical monitor).
    channel_max = np.max(rgb, axis=2)
    channel_min = np.min(rgb, axis=2)
    color_spread = channel_max - channel_min

    mean_color_spread = float(np.mean(color_spread))
    colorful_pixel_ratio = float(np.mean(color_spread > 0.16))

    if mean_color_spread > 0.085 or colorful_pixel_ratio > 0.075:
        return False, WRONG_IMAGE_MESSAGE

    # 2. Global Intensity & Contrast Check:
    gray = np.mean(rgb, axis=2)
    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))

    # Reject overly dark/blank or overly bright/white-background images (charts, documents, screenshots)
    if mean_intensity < 0.07 or mean_intensity > 0.52:
        return False, WRONG_IMAGE_MESSAGE

    if std_intensity < 0.11 or std_intensity > 0.43:
        return False, WRONG_IMAGE_MESSAGE

    dark_ratio = float(np.mean(gray < 0.28))
    mid_ratio = float(np.mean((gray >= 0.28) & (gray <= 0.85)))
    bright_ratio = float(np.mean(gray > 0.85))

    # Chest CT scans have substantial dark air (background + lungs) and gray soft tissue,
    # with only small amounts of bright bone/calcification.
    if dark_ratio < 0.25 or dark_ratio > 0.88:
        return False, WRONG_IMAGE_MESSAGE

    if mid_ratio < 0.12:
        return False, WRONG_IMAGE_MESSAGE

    if bright_ratio > 0.22:
        return False, WRONG_IMAGE_MESSAGE

    # 3. Axial Chest CT Anatomical Geometry Check:
    # A) Outer corners (surrounding air outside the patient torso) must be dark
    c_h, c_w = max(4, int(h * 0.12)), max(4, int(w * 0.12))
    corners = np.concatenate(
        [
            gray[:c_h, :c_w].ravel(),
            gray[:c_h, w - c_w :].ravel(),
            gray[h - c_h :, :c_w].ravel(),
            gray[h - c_h :, w - c_w :].ravel(),
        ]
    )
    corners_mean = float(np.mean(corners))
    center_torso = gray[int(h * 0.22) : int(h * 0.78), int(w * 0.18) : int(w * 0.82)]
    center_mean = float(np.mean(center_torso))

    if corners_mean > 0.28 or (center_mean - corners_mean) < 0.07:
        return False, WRONG_IMAGE_MESSAGE

    # B) Bilateral Lung Fields vs Central Mediastinum/Spine:
    # In an axial Lung CT scan, the vertical midline (heart, mediastinum, vertebra)
    # sits between two dark air-filled lung cavities (left and right lung fields).
    midline = gray[int(h * 0.25) : int(h * 0.78), int(w * 0.43) : int(w * 0.57)]
    left_lung = gray[int(h * 0.30) : int(h * 0.70), int(w * 0.23) : int(w * 0.42)]
    right_lung = gray[int(h * 0.30) : int(h * 0.70), int(w * 0.58) : int(w * 0.77)]

    midline_mean = float(np.mean(midline))
    left_lung_mean = float(np.mean(left_lung))
    right_lung_mean = float(np.mean(right_lung))
    lungs_mean = 0.5 * (left_lung_mean + right_lung_mean)

    left_lung_dark_fraction = float(np.mean(left_lung < 0.32))
    right_lung_dark_fraction = float(np.mean(right_lung < 0.32))

    # Mediastinum/spine must contain tissue and be brighter than the bilateral air-filled lung cavities
    if midline_mean < 0.20:
        return False, WRONG_IMAGE_MESSAGE

    if (midline_mean - lungs_mean) < 0.04:
        return False, WRONG_IMAGE_MESSAGE

    if left_lung_dark_fraction < 0.22 or right_lung_dark_fraction < 0.22:
        return False, WRONG_IMAGE_MESSAGE

    return True, "Valid Lung CT Scan"
