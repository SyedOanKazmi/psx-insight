"""
JWT authentication helpers for Invock Investments.

Issues signed JWT access tokens on login/register and provides FastAPI
dependencies that read the `Authorization: Bearer <token>` header, validate the
token, and load the current user (with optional role checks).
"""
import os
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from db import get_db

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me-0123456789abcdef-invock")
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day
PBKDF2_ROUNDS = 200_000

bearer = HTTPBearer(auto_error=True)


# ─── Passwords ────────────────────────────────────────────────────────────────
def hash_password(pw: str) -> str:
    """Salted PBKDF2-HMAC-SHA256 hash, stored as pbkdf2$<salt>$<hash>."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), PBKDF2_ROUNDS).hex()
    return f"pbkdf2${salt}${digest}"


def verify_password(pw: str, stored: str) -> bool:
    """Accept both PBKDF2 (real users) and the seeded sha256 demo hashes."""
    if stored.startswith("pbkdf2$"):
        _, salt, digest = stored.split("$")
        check = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), PBKDF2_ROUNDS).hex()
        return hmac.compare_digest(check, digest)
    return hmac.compare_digest(hashlib.sha256(pw.encode()).hexdigest(), stored)


# ─── Tokens ───────────────────────────────────────────────────────────────────
def create_access_token(email: str, role: str, name: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    payload = {"sub": email, "role": role, "name": name, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    """Decode the bearer token and return the user record from the database."""
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or expired token")
    conn = get_db()
    user = conn.execute("SELECT email, role, name FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return dict(user)


def require_roles(*roles):
    """Dependency factory: allow only the given roles."""
    def checker(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
        return user
    return checker
