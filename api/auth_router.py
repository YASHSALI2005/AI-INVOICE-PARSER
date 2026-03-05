"""Auth routes: send OTP, verify OTP (phone login), me, logout."""
import random
import string
import time
from datetime import datetime, timedelta
from typing import Dict, Optional
from urllib.parse import quote

import requests
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
import re
from sqlalchemy.orm import Session

from config import get_settings
from db.models import User, Session as SessionModel
from api.deps import get_db, get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])

# In-memory OTP store: { phone: { "otp": str, "expires_at": float } }
_otp_store: Dict[str, dict] = {}
OTP_TTL_SECONDS = 300  # 5 minutes
SESSION_TTL_DAYS = 30


def _generate_otp(length: int = 6) -> str:
    return "".join(random.choices(string.digits, k=length))


def _send_sms(mobile_number: str, otp: str) -> bool:
    """Call SMS gateway (same logic as your sendOtp)."""
    settings = get_settings()
    if not settings.sms_api_url or not settings.sms_api_key:
        return False
    message = f"Dear User, your OTP for registration with Retello is {otp}."
    encoded_message = quote(message)
    url = (
        f"{settings.sms_api_url}"
        f"?username={settings.sms_api_username}"
        f"&apikey={settings.sms_api_key}"
        f"&senderid={settings.sms_sender_id}"
        f"&route={settings.sms_route}"
        f"&mobile={mobile_number}"
        f"&text={encoded_message}"
    )
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return False
        text = str(resp.text or resp.content).strip().lower()
        return "message submitted successfully" in text
    except Exception:
        return False


class SendOtpBody(BaseModel):
    mobileNumber: str


class VerifyOtpBody(BaseModel):
    mobileNumber: str
    otp: str


class UpdateProfileBody(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None


@router.post("/send-otp")
def send_otp(body: SendOtpBody):
    """Request OTP for the given phone number. Sends SMS via configured gateway."""
    phone = (body.mobileNumber or "").strip()
    if not phone or len(phone) < 10:
        raise HTTPException(status_code=400, detail="Invalid phone number")
    otp = _generate_otp(6)
    _otp_store[phone] = {"otp": otp, "expires_at": time.time() + OTP_TTL_SECONDS}
    ok = _send_sms(phone, otp)
    if not ok:
        raise HTTPException(status_code=502, detail="Failed to send SMS")
    return {"success": True, "message": "OTP sent"}


@router.post("/verify-otp")
def verify_otp(body: VerifyOtpBody, db: Session = Depends(get_db)):
    """Verify OTP, create or get user, create session, return token."""
    phone = (body.mobileNumber or "").strip()
    otp = (body.otp or "").strip()
    if not phone or not otp:
        raise HTTPException(status_code=400, detail="Missing phone or OTP")
    entry = _otp_store.get(phone)
    if not entry:
        raise HTTPException(status_code=400, detail="OTP expired or not requested")
    if time.time() > entry["expires_at"]:
        del _otp_store[phone]
        raise HTTPException(status_code=400, detail="OTP expired")
    if entry["otp"] != otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    del _otp_store[phone]

    # Create or get user
    user = db.query(User).filter(User.phone == phone).first()
    if not user:
        user = User(phone=phone)
        db.add(user)
        db.flush()
    user.last_login_at = datetime.utcnow()

    # Create session
    token = "".join(random.choices(string.ascii_letters + string.digits, k=32))
    expires_at = datetime.utcnow() + timedelta(days=SESSION_TTL_DAYS)
    session = SessionModel(token=token, user_id=user.id, expires_at=expires_at)
    db.add(session)
    db.commit()

    return {"success": True, "token": token, "user": {"id": user.id, "phone": user.phone}}


@router.get("/me")
def auth_me(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Return current user. Requires Authorization: Bearer <token>."""
    return {
        "id": user.id,
        "phone": user.phone,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@router.put("/me")
def update_me(
    body: UpdateProfileBody,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Update basic profile fields for the current user."""
    if body.email is not None:
        email = body.email.strip()
        if email and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            raise HTTPException(status_code=400, detail="Invalid email address")
        # Enforce unique email if provided
        if email:
            existing = (
                db.query(User)
                .filter(User.email == email, User.id != user.id)
                .first()
            )
            if existing:
                raise HTTPException(status_code=400, detail="Email already in use")
            user.email = email
        else:
            user.email = None

    if body.first_name is not None:
        user.first_name = body.first_name.strip() or None
    if body.last_name is not None:
        user.last_name = body.last_name.strip() or None

    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "id": user.id,
        "phone": user.phone,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@router.post("/logout")
def logout(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    db: Session = Depends(get_db),
):
    """Invalidate current session. Requires Authorization: Bearer <token>."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
        db.query(SessionModel).filter(SessionModel.token == token).delete()
        db.commit()
    return {"success": True}
