"""SQLAlchemy models for PostgreSQL."""
from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, DateTime, Text, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from .base import Base


class Extraction(Base):
    """Stored invoice extraction (LLM or hybrid)."""
    __tablename__ = "extractions"

    id = Column(String(24), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    vendor_name = Column(String(512), nullable=False, index=True)
    vendor_fingerprint = Column(String(512), nullable=False, index=True)
    data = Column(JSONB, nullable=False)
    confidence = Column(Float, default=1.0)
    source = Column(String(32), default="llm")  # llm, template, hybrid
    image_hash = Column(String(64), nullable=True)
    verified = Column(Boolean, default=False)

    corrections = relationship("Correction", back_populates="extraction", order_by="Correction.created_at")


class Correction(Base):
    """User correction for an extraction."""
    __tablename__ = "corrections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    extraction_id = Column(String(24), ForeignKey("extractions.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    original_data = Column(JSONB, nullable=False)
    corrected_data = Column(JSONB, nullable=False)
    fields_changed = Column(JSONB, nullable=False)  # list of field names

    extraction = relationship("Extraction", back_populates="corrections")


class Vendor(Base):
    """Vendor index for quick lookup and stats."""
    __tablename__ = "vendors"

    fingerprint = Column(String(512), primary_key=True)
    name = Column(String(512), nullable=False)
    extraction_count = Column(Integer, default=0)
    template_exists = Column(Boolean, default=False)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Template(Base):
    """Invoice2data template content per vendor."""
    __tablename__ = "templates"

    vendor_fingerprint = Column(String(512), primary_key=True)
    content = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class VendorFingerprint(Base):
    """Mapping vendor_fingerprint -> extraction_id for get_vendor_extractions."""
    __tablename__ = "vendor_fingerprints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vendor_fingerprint = Column(String(512), nullable=False, index=True)
    extraction_id = Column(String(24), nullable=False, index=True)


class User(Base):
    """Authenticated user (phone OTP login). Uses auth_users to avoid conflict with existing users table."""
    __tablename__ = "auth_users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    phone = Column(String(32), unique=True, nullable=False, index=True)
    first_name = Column(String(128), nullable=True)
    last_name = Column(String(128), nullable=True)
    email = Column(String(255), unique=True, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Session(Base):
    """Active session (token -> user). Uses auth_sessions to avoid conflict with existing sessions table."""
    __tablename__ = "auth_sessions"

    token = Column(String(64), primary_key=True)
    user_id = Column(Integer, ForeignKey("auth_users.id", ondelete="CASCADE"), nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class AiInvoice(Base):
    """AI-parsed invoice record (normalized fields + optional link to extraction)."""
    __tablename__ = "ai_invoice"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("auth_users.id", ondelete="CASCADE"), nullable=True, index=True)
    extraction_id = Column(String(24), ForeignKey("extractions.id", ondelete="SET NULL"), nullable=True, index=True)
    vendor_name = Column(String(512), nullable=True, index=True)
    invoice_number = Column(String(128), nullable=True, index=True)
    document_type = Column(String(64), nullable=True)  # e.g. Sales Invoice bill, Purchase Order bill
    invoice_date = Column(DateTime, nullable=True)
    total_amount = Column(Float, nullable=True)
    currency = Column(String(16), nullable=True)
    raw_data = Column(JSONB, nullable=True)  # full extracted JSON
    status = Column(String(32), default="pending")  # e.g. pending, verified, rejected
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class InvoiceAudit(Base):
    """Saved audit comparing a final invoice with one or more supporting invoices."""
    __tablename__ = "invoice_audits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("auth_users.id", ondelete="CASCADE"), nullable=False, index=True)
    final_invoice_id = Column(Integer, ForeignKey("ai_invoice.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class InvoiceAuditItem(Base):
    """Link table: each row associates one supporting invoice with an audit."""
    __tablename__ = "invoice_audit_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    audit_id = Column(Integer, ForeignKey("invoice_audits.id", ondelete="CASCADE"), nullable=False, index=True)
    invoice_id = Column(Integer, ForeignKey("ai_invoice.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
