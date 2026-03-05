"""
PostgreSQL implementation of the invoice DB interface.
Same API as the original InvoiceDatabase (save_extraction, get_extraction, save_correction, etc.).
"""
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from .database import SessionLocal
from .models import Extraction, Correction, Vendor, Template, VendorFingerprint


def _vendor_fingerprint(vendor_name: str) -> str:
    if not vendor_name:
        return "unknown"
    normalized = vendor_name.lower().strip()
    for suffix in [
        ", inc.", ", inc", " inc.", " inc", ", llc", " llc", ", ltd", " ltd",
        " limited", " corporation", " corp", " corp.",
    ]:
        normalized = normalized.replace(suffix, "")
    return normalized.strip()


def _extraction_id(data: Dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]


class PostgresInvoiceRepository:
    """Invoice database backed by PostgreSQL. Same interface as original InvoiceDatabase."""

    def save_extraction(
        self,
        extracted_data: Dict[str, Any],
        confidence: float = 1.0,
        source: str = "llm",
        image_hash: Optional[str] = None,
    ) -> str:
        extraction_id = _extraction_id(extracted_data)
        vendor_name = extracted_data.get("vendor_name", "Unknown")
        vendor_fp = _vendor_fingerprint(vendor_name)

        db = SessionLocal()
        try:
            existing = db.query(Extraction).filter(Extraction.id == extraction_id).first()
            if existing:
                return extraction_id

            extraction = Extraction(
                id=extraction_id,
                vendor_name=vendor_name,
                vendor_fingerprint=vendor_fp,
                data=extracted_data,
                confidence=confidence,
                source=source,
                image_hash=image_hash,
                verified=False,
            )
            db.add(extraction)

            vendor = db.query(Vendor).filter(Vendor.fingerprint == vendor_fp).first()
            if not vendor:
                vendor = Vendor(
                    fingerprint=vendor_fp,
                    name=vendor_name,
                    extraction_count=0,
                    template_exists=False,
                )
                db.add(vendor)
            vendor.extraction_count += 1
            vendor.last_seen = datetime.utcnow()

            fp_link = VendorFingerprint(vendor_fingerprint=vendor_fp, extraction_id=extraction_id)
            db.add(fp_link)

            db.commit()
        finally:
            db.close()

        return extraction_id

    def list_extractions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent extractions (id, vendor_name, created_at, confidence, summary fields)."""
        db = SessionLocal()
        try:
            rows = (
                db.query(Extraction)
                .order_by(Extraction.created_at.desc())
                .limit(limit)
                .all()
            )
            out = []
            for row in rows:
                data = row.data or {}
                summary = data.get("summary") or {}
                total = (
                    data.get("total_amount")
                    or data.get("total_amount_due")
                    or summary.get("order_total")
                    or summary.get("total_amount_due")
                    or summary.get("total_commitment")
                )
                out.append({
                    "id": row.id,
                    "vendor_name": row.vendor_name,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "confidence": row.confidence,
                    "verified": row.verified,
                    "invoice_number": data.get("invoice_number") or data.get("po_number") or data.get("order_number"),
                    "total_amount": total,
                    "currency": data.get("currency"),
                })
            return out
        finally:
            db.close()

    def get_extraction(self, extraction_id: str) -> Optional[Dict[str, Any]]:
        db = SessionLocal()
        try:
            row = db.query(Extraction).filter(Extraction.id == extraction_id).first()
            if not row:
                return None
            out = {
                "id": row.id,
                "timestamp": row.created_at.isoformat() if row.created_at else None,
                "vendor_name": row.vendor_name,
                "vendor_fingerprint": row.vendor_fingerprint,
                "data": row.data,
                "confidence": row.confidence,
                "source": row.source,
                "image_hash": row.image_hash,
                "corrections": [],
                "verified": row.verified,
            }
            for c in row.corrections:
                out["corrections"].append({
                    "timestamp": c.created_at.isoformat() if c.created_at else None,
                    "original": c.original_data,
                    "corrected": c.corrected_data,
                    "fields_changed": c.fields_changed,
                })
            return out
        finally:
            db.close()

    def save_correction(
        self,
        extraction_id: str,
        original_data: Dict[str, Any],
        corrected_data: Dict[str, Any],
        corrected_fields: List[str],
    ) -> bool:
        db = SessionLocal()
        try:
            extraction = db.query(Extraction).filter(Extraction.id == extraction_id).first()
            if not extraction:
                return False
            correction = Correction(
                extraction_id=extraction_id,
                original_data=original_data,
                corrected_data=corrected_data,
                fields_changed=corrected_fields,
            )
            db.add(correction)
            extraction.verified = True
            extraction.data = corrected_data
            db.commit()
            return True
        finally:
            db.close()

    def get_vendor_extractions(self, vendor_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        vendor_fp = _vendor_fingerprint(vendor_name)
        db = SessionLocal()
        try:
            links = (
                db.query(VendorFingerprint)
                .filter(VendorFingerprint.vendor_fingerprint == vendor_fp)
                .order_by(VendorFingerprint.id.desc())
                .limit(limit)
                .all()
            )
            extraction_ids = [l.extraction_id for l in links]
            out = []
            for eid in extraction_ids:
                ext = self.get_extraction(eid)
                if ext:
                    out.append(ext)
            return out
        finally:
            db.close()

    def has_template(self, vendor_name: str) -> bool:
        vendor_fp = _vendor_fingerprint(vendor_name)
        db = SessionLocal()
        try:
            return db.query(Template).filter(Template.vendor_fingerprint == vendor_fp).first() is not None
        finally:
            db.close()

    def save_template(self, vendor_name: str, template_content: str) -> str:
        vendor_fp = _vendor_fingerprint(vendor_name)
        db = SessionLocal()
        try:
            t = db.query(Template).filter(Template.vendor_fingerprint == vendor_fp).first()
            if t:
                t.content = template_content
            else:
                t = Template(vendor_fingerprint=vendor_fp, content=template_content)
                db.add(t)
            v = db.query(Vendor).filter(Vendor.fingerprint == vendor_fp).first()
            if v:
                v.template_exists = True
            db.commit()
            return f"{vendor_fp}.yml"
        finally:
            db.close()

    def get_template(self, vendor_name: str) -> Optional[str]:
        vendor_fp = _vendor_fingerprint(vendor_name)
        db = SessionLocal()
        try:
            t = db.query(Template).filter(Template.vendor_fingerprint == vendor_fp).first()
            return t.content if t else None
        finally:
            db.close()

    def get_all_vendors(self) -> List[Dict[str, Any]]:
        db = SessionLocal()
        try:
            rows = db.query(Vendor).order_by(Vendor.extraction_count.desc()).all()
            return [
                {
                    "fingerprint": r.fingerprint,
                    "name": r.name,
                    "extraction_count": r.extraction_count,
                    "template_exists": r.template_exists,
                    "first_seen": r.first_seen.isoformat() if r.first_seen else None,
                    "last_seen": r.last_seen.isoformat() if r.last_seen else None,
                }
                for r in rows
            ]
        finally:
            db.close()

    def get_stats(self) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            total_extractions = db.query(Extraction).count()
            total_templates = db.query(Template).count()
            total_corrections = db.query(Correction).count()
            total_vendors = db.query(Vendor).count()
            return {
                "total_extractions": total_extractions,
                "total_templates": total_templates,
                "total_corrections": total_corrections,
                "total_vendors": total_vendors,
            }
        finally:
            db.close()
