import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "triage_users.db")

# Medical roles matching the Android frontend dropdown in RegisterActivity.kt
MEDICAL_ROLES = [
    "Senior Radiologist",
    "Thoracic Oncologist",
    "Pulmonologist",
    "Resident Radiologist",
    "Emergency Physician",
    "Medical Officer",
    "Radiologic Technologist",
]

# Default demo doctor account matching AuthRepository.kt in the frontend
DEFAULT_DEMO_EMAIL = "doctor@hospital.com"
DEFAULT_DEMO_PASS = "password123"
DEFAULT_DEMO_NAME = "Dr. Alexander Wright"
DEFAULT_DEMO_ROLE = "Senior Radiologist"

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def get_current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str, salt: Optional[str] = None) -> str:
    if salt is None:
        salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()
    return f"{salt}${pwd_hash}"


def verify_password(password: str, stored_hash: str) -> bool:
    if "$" not in stored_hash:
        return False
    salt, _ = stored_hash.split("$", 1)
    expected_hash = hash_password(password, salt)
    return hmac.compare_digest(expected_hash, stored_hash)


def derive_doctor_name_from_email(email: str) -> str:
    """
    Derives a default clinician name from the email address if full_name is not provided,
    matching AuthRepository.kt behavior in the Android frontend.
    """
    username_part = email.split("@")[0].strip()
    cleaned = re.sub(r"[._-]+", " ", username_part).strip()
    if not cleaned or cleaned.lower() in ("admin", "doctor"):
        return DEFAULT_DEMO_NAME
    titled = " ".join(word.capitalize() for word in cleaned.split())
    if titled.lower().startswith("dr.") or titled.lower().startswith("dr "):
        return titled
    return f"Dr. {titled}"


def init_db() -> None:
    """
    Initializes SQLite tables for users and active sessions, and seeds the default
    demo doctor account if no users exist yet.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'Senior Radiologist',
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                email TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """
        )
        conn.commit()

        # Ensure default demo account exists (doctor@hospital.com / password123)
        cursor.execute(
            "SELECT id FROM users WHERE LOWER(email) = LOWER(?)",
            (DEFAULT_DEMO_EMAIL,),
        )
        if cursor.fetchone() is None:
            cursor.execute(
                """
                INSERT INTO users (email, password_hash, full_name, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    DEFAULT_DEMO_EMAIL.lower(),
                    hash_password(DEFAULT_DEMO_PASS),
                    DEFAULT_DEMO_NAME,
                    DEFAULT_DEMO_ROLE,
                    get_current_timestamp(),
                ),
            )
            conn.commit()


def validate_email_format(email: str) -> bool:
    return bool(EMAIL_REGEX.match(email.strip()))


def create_session_token(user_id: int, email: str) -> str:
    token = secrets.token_urlsafe(32)
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO sessions (token, user_id, email, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (token, user_id, email.lower(), get_current_timestamp()),
        )
        conn.commit()
    return token


def register_doctor(
    email: str,
    new_password: str,
    password_confirmation: Optional[str] = None,
    full_name: Optional[str] = None,
    role: Optional[str] = None,
) -> Tuple[bool, int, str, Optional[Dict[str, Any]], Optional[str]]:
    """
    Registers a new doctor account following the frontend RegisterActivity.kt validation rules.
    Returns (success, http_status, message, user_dict, session_token).
    """
    email_clean = (email or "").strip().lower()
    password_clean = (new_password or "").strip()
    confirm_clean = (
        password_confirmation.strip()
        if password_confirmation is not None
        else password_clean
    )

    if not email_clean:
        return False, 400, "Please enter a valid email address", None, None

    if not validate_email_format(email_clean):
        return False, 400, "Please enter a valid email format", None, None

    if not password_clean:
        return False, 400, "Please enter a password", None, None

    if len(password_clean) < 6:
        return False, 400, "Password must be at least 6 characters", None, None

    if password_clean != confirm_clean:
        return False, 400, "Passwords do not match", None, None

    resolved_name = (full_name or "").strip()
    if not resolved_name:
        resolved_name = derive_doctor_name_from_email(email_clean)

    resolved_role = (role or "").strip()
    if not resolved_role:
        resolved_role = "Senior Radiologist"

    created_at = get_current_timestamp()
    pwd_hash = hash_password(password_clean)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM users WHERE LOWER(email) = LOWER(?)",
            (email_clean,),
        )
        if cursor.fetchone() is not None:
            return (
                False,
                409,
                "An account with this email already exists.",
                None,
                None,
            )

        cursor.execute(
            """
            INSERT INTO users (email, password_hash, full_name, role, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (email_clean, pwd_hash, resolved_name, resolved_role, created_at),
        )
        user_id = cursor.lastrowid
        conn.commit()

    token = create_session_token(user_id, email_clean)
    user_data = {
        "id": user_id,
        "email": email_clean,
        "full_name": resolved_name,
        "role": resolved_role,
        "created_at": created_at,
    }
    return True, 201, "Account created successfully!", user_data, token


def authenticate_doctor(
    email: str, password: str
) -> Tuple[bool, int, str, Optional[Dict[str, Any]], Optional[str]]:
    """
    Authenticates a doctor with email and password matching LoginActivity.kt & AuthRepository.kt.
    Returns (success, http_status, message, user_dict, session_token).
    """
    email_clean = (email or "").strip().lower()
    password_clean = (password or "").strip()

    if not email_clean:
        return False, 400, "Please enter your email address", None, None

    if not password_clean:
        return False, 400, "Please enter your password", None, None

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, email, password_hash, full_name, role, created_at
            FROM users
            WHERE LOWER(email) = LOWER(?)
            """,
            (email_clean,),
        )
        row = cursor.fetchone()

    if row is None or not verify_password(password_clean, row["password_hash"]):
        return False, 401, "Invalid doctor email or password.", None, None

    user_data = {
        "id": row["id"],
        "email": row["email"],
        "full_name": row["full_name"],
        "role": row["role"],
        "created_at": row["created_at"],
    }
    token = create_session_token(row["id"], row["email"])
    return (
        True,
        200,
        f"Welcome back, {row['full_name']}!",
        user_data,
        token,
    )


def update_doctor_password(
    email: str, new_password: str, password_confirmation: str
) -> Tuple[bool, int, str]:
    """
    Updates/resets a doctor's password given doctor_email, new_password, and password_confirmation.
    """
    email_clean = (email or "").strip().lower()
    password_clean = (new_password or "").strip()
    confirm_clean = (password_confirmation or "").strip()

    if not email_clean or not validate_email_format(email_clean):
        return False, 400, "Please enter a valid email address"

    if not password_clean:
        return False, 400, "Please enter a new password"

    if len(password_clean) < 6:
        return False, 400, "Password must be at least 6 characters"

    if password_clean != confirm_clean:
        return False, 400, "Passwords do not match"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, full_name FROM users WHERE LOWER(email) = LOWER(?)",
            (email_clean,),
        )
        row = cursor.fetchone()
        if row is None:
            return False, 404, "No account found with this doctor email."

        new_hash = hash_password(password_clean)
        cursor.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (new_hash, row["id"]),
        )
        conn.commit()

    return True, 200, "Password updated successfully!"


def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    token_clean = (token or "").strip()
    if not token_clean:
        return None
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT u.id, u.email, u.full_name, u.role, u.created_at
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ?
            """,
            (token_clean,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    email_clean = (email or "").strip().lower()
    if not email_clean:
        return None
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, email, full_name, role, created_at
            FROM users
            WHERE LOWER(email) = LOWER(?)
            """,
            (email_clean,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)


def revoke_session_token(token: str) -> bool:
    token_clean = (token or "").strip()
    if not token_clean:
        return False
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE token = ?", (token_clean,))
        conn.commit()
        return cursor.rowcount > 0


def list_all_doctors() -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, email, full_name, role, created_at FROM users ORDER BY id ASC"
        )
        return [dict(row) for row in cursor.fetchall()]
