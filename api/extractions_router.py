"""Extraction and learning API: upload, extract, corrections, vendors, stats. All require auth."""
import hashlib
import tempfile
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import extractor
from confidence_scorer import calculate_confidence
from invoice_db import get_database
from template_generator import generate_template, should_generate_template
from idfy_client import verify_gstin, verify_msme, verify_bank_account

from api.deps import get_current_user
from db.database import SessionLocal
from db.models import AiInvoice, User

try:
    import hybrid_extractor
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

router = APIRouter(prefix="/api", tags=["extractions"])


def _get_api_key(provider: str) -> str:
    from config import get_settings
    s = get_settings()
    if provider == "Gemini":
        return s.gemini_api_key or ""
    if provider == "OpenAI":
        return s.openai_api_key or ""
    if provider == "Claude":
        return s.claude_api_key or ""
    return ""


def _enrich_with_idfy_checks(data: Dict[str, Any]) -> None:
    """
    For extracted invoice payload `data`, call IDfy (if configured) to validate
    GSTIN, MSME registration, and bank account + IFSC, and attach status fields.

    Status fields are:
      - gstin_status: 'valid' | 'invalid' | 'unknown' | 'error'
      - msme_status: same
      - bank_account_status: same
      - ifsc_status: same (mirrors bank_account_status when both are checked)
    """
    # GSTIN
    gstin = data.get("gstin")
    if isinstance(gstin, str) and gstin.strip():
        data["gstin_status"] = verify_gstin(gstin.strip())

    # MSME registration
    msme = (
        data.get("msme")
        or data.get("MSME")
        or data.get("msme_number")
    )
    if isinstance(msme, str) and msme.strip():
        data["msme_status"] = verify_msme(msme.strip())

    # Bank account + IFSC
    account = (
        data.get("bank_account_number")
        or data.get("account_number")
        or data.get("account_no")
    )
    ifsc = data.get("ifsc") or data.get("IFSC") or data.get("ifsc_code")

    if isinstance(account, (str, int)) and isinstance(ifsc, str):
        account_str = str(account).strip()
        ifsc_str = ifsc.strip()
        if account_str and ifsc_str:
            status = verify_bank_account(account_str, ifsc_str)
            data["bank_account_status"] = status
            data["ifsc_status"] = status


def _perform_extraction_from_bytes(
    content: bytes,
    content_type: str,
    provider: str,
    model_name: str,
    document_type: str,
    use_hybrid: bool,
) -> Dict[str, Any]:
    """
    Core extraction logic shared by the sync endpoint and background jobs.

    Returns a dict with keys: data, confidence, extraction_id, error.
    """
    import os

    api_key = _get_api_key(provider)
    if not api_key:
        return {
            "error": f"Missing API key for {provider}. Set in .env.",
            "data": None,
            "confidence": None,
            "extraction_id": None,
        }

    if not model_name:
        model_name = (
            "gemini-2.5-flash"
            if provider == "Gemini"
            else "gpt-4o-mini"
            if provider == "OpenAI"
            else "claude-3-haiku-20240307"
        )

    suffix = ".pdf" if "pdf" in (content_type or "").lower() else ".png"

    images = None
    if "pdf" in (content_type or "").lower():
        try:
            images = extractor.convert_pdf_to_images(content)
        except RuntimeError as e:
            return {
                "error": str(e),
                "data": None,
                "confidence": None,
                "extraction_id": None,
            }
        if not images:
            return {
                "error": "Failed to convert PDF to images (no pages).",
                "data": None,
                "confidence": None,
                "extraction_id": None,
            }
    else:
        from PIL import Image
        import io

        images = [Image.open(io.BytesIO(content))]

    if use_hybrid and HYBRID_AVAILABLE:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            data = hybrid_extractor.hybrid_extract_invoice(
                tmp_path,
                images,
                api_key,
                provider=provider,
                model_name=model_name,
                document_type=document_type,
            )
            display_data = data.get("final_data", data)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        data = extractor.extract_invoice_data(
            images,
            api_key,
            provider=provider,
            model_name=model_name,
            document_type=document_type,
        )
        display_data = data if isinstance(data, dict) else {}

    # Persist the selected document_type alongside extracted data so it is
    # available in downstream UIs (e.g. edit screen bill type).
    if isinstance(display_data, dict) and document_type:
        display_data.setdefault("document_type", document_type)

    # Enrich with IDfy verification statuses for GSTIN / MSME / bank account + IFSC.
    if isinstance(display_data, dict):
        try:
            _enrich_with_idfy_checks(display_data)
        except Exception:
            # If IDfy is misconfigured or unavailable, we keep extraction data as-is.
            pass

    if "error" in display_data:
        return {
            "error": display_data.get("error"),
            "data": display_data,
            "confidence": None,
            "extraction_id": None,
        }

    confidence_result = calculate_confidence(display_data)
    db = get_database()
    img_bytes = images[0].tobytes() if images else b""
    image_hash = hashlib.md5(img_bytes[:10000]).hexdigest()
    extraction_id = db.save_extraction(
        display_data,
        confidence=confidence_result["overall"],
        source="hybrid" if (use_hybrid and HYBRID_AVAILABLE) else "llm",
        image_hash=image_hash,
    )
    if should_generate_template(display_data, confidence_result["overall"]):
        vendor_name = display_data.get("vendor_name")
        if vendor_name and not db.has_template(vendor_name):
            template = generate_template(vendor_name, display_data)
            db.save_template(vendor_name, template)

    return {
        "error": None,
        "data": display_data,
        "confidence": confidence_result,
        "extraction_id": extraction_id,
    }


@router.post("/extract")
async def extract(
    file: UploadFile = File(...),
    provider: str = Form("Gemini"),
    model_name: str = Form(""),
    document_type: str = Form("Sales Invoice bill"),
    use_hybrid: bool = Form(True),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Upload an invoice (PDF or image) and extract data using LLM (and optional hybrid).
    Returns extraction result, confidence, and extraction_id if saved.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file")

    content = await file.read()
    result = _perform_extraction_from_bytes(
        content=content,
        content_type=file.content_type or "",
        provider=provider,
        model_name=model_name,
        document_type=document_type,
        use_hybrid=use_hybrid,
    )

    if result["error"]:
        return {"error": result["error"], "extraction_id": None, "confidence": None}

    return {
        "data": result["data"],
        "confidence": result["confidence"],
        "extraction_id": result["extraction_id"],
    }


class CorrectionBody(BaseModel):
    original_data: Dict[str, Any]
    corrected_data: Dict[str, Any]
    corrected_fields: List[str]


@router.get("/extractions")
def list_extractions(
    limit: int = 50,
    user: User = Depends(get_current_user),
):
    """List recent extractions (summary only)."""
    db = get_database()
    return db.list_extractions(limit=min(limit, 100))


@router.get("/extractions/{extraction_id}")
def get_extraction(extraction_id: str, user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Get a stored extraction by ID."""
    db = get_database()
    ext = db.get_extraction(extraction_id)
    if not ext:
        raise HTTPException(status_code=404, detail="Extraction not found")
    return ext


@router.post("/extractions/{extraction_id}/corrections")
def save_correction(extraction_id: str, body: CorrectionBody, user: User = Depends(get_current_user)):
    """Save a user correction for an extraction."""
    db = get_database()
    ok = db.save_correction(
        extraction_id,
        body.original_data,
        body.corrected_data,
        body.corrected_fields,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Extraction not found")
    # Also update AiInvoice document_type if this extraction is linked to an AiInvoice row
    # and the user edited the bill/document type field.
    if "document_type" in body.corrected_fields:
        from db.database import SessionLocal  # local import to avoid circulars at import time
        from db.models import AiInvoice

        session = SessionLocal()
        try:
            invoice = (
                session.query(AiInvoice)
                .filter(AiInvoice.extraction_id == extraction_id, AiInvoice.user_id == user.id)
                .first()
            )
            if invoice is not None:
                invoice.document_type = body.corrected_data.get("document_type")  # type: ignore[assignment]
                invoice.updated_at = datetime.utcnow()
                session.add(invoice)
                session.commit()
        finally:
            session.close()

    return {"success": True}


@router.get("/vendors")
def list_vendors(user: User = Depends(get_current_user)):
    """List all known vendors (from extractions)."""
    db = get_database()
    return db.get_all_vendors()


@router.get("/stats")
def stats(user: User = Depends(get_current_user)):
    """Database stats: extractions, templates, corrections, vendors."""
    db = get_database()
    return db.get_stats()


@router.get("/dashboard/stats")
def dashboard_stats(user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Aggregated dashboard metrics for the current user."""
    db_session = SessionLocal()
    try:
        now = datetime.utcnow()
        cutoff = now - timedelta(days=30)

        base_q = (
            db_session.query(AiInvoice)
            .filter(
                AiInvoice.user_id == user.id,
                AiInvoice.status != "duplicate",
            )
        )

        total_invoices = base_q.count()
        invoices_last_30d = (
            base_q.filter(AiInvoice.created_at >= cutoff).count()
        )
        failed_invoices = base_q.filter(AiInvoice.status == "error").count()
        pending_invoices = (
            base_q.filter(AiInvoice.status.in_(["pending", "in_progress"])).count()
        )

        credit_q = base_q.filter(
            AiInvoice.document_type == "Credit bill",
            AiInvoice.created_at >= cutoff,
        )
        debit_q = base_q.filter(
            AiInvoice.document_type == "Debit bill",
            AiInvoice.created_at >= cutoff,
        )

        credit_total_30d = sum(row.total_amount or 0 for row in credit_q.all())
        debit_total_30d = sum(row.total_amount or 0 for row in debit_q.all())
        net_cash_flow_30d = credit_total_30d - debit_total_30d

        from db.models import InvoiceAudit  # local import to avoid circular import at module load

        audits_q = db_session.query(InvoiceAudit).filter(
            InvoiceAudit.user_id == user.id
        )
        total_audits = audits_q.count()
        audits_last_30d = audits_q.filter(InvoiceAudit.created_at >= cutoff).count()

        return {
            "total_invoices": total_invoices,
            "invoices_last_30d": invoices_last_30d,
            "failed_invoices": failed_invoices,
            "pending_invoices": pending_invoices,
            "credit_total_30d": credit_total_30d,
            "debit_total_30d": debit_total_30d,
            "net_cash_flow_30d": net_cash_flow_30d,
            "total_audits": total_audits,
            "audits_last_30d": audits_last_30d,
        }
    finally:
        db_session.close()


def _update_ai_invoice_status(
    ai_invoice_id: int,
    status: str,
    extraction_payload: Optional[Dict[str, Any]] = None,
    extraction_id: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Small helper for background jobs to update AiInvoice rows."""
    from datetime import datetime

    db_session = SessionLocal()
    try:
        invoice = db_session.query(AiInvoice).filter(AiInvoice.id == ai_invoice_id).first()
        if not invoice:
            return

        invoice.status = status
        invoice.updated_at = datetime.utcnow()

        if error:
            invoice.raw_data = {"error": error}

        if extraction_payload is not None:
            data = extraction_payload or {}
            summary = data.get("summary") or {}
            total = (
                data.get("total_amount")
                or data.get("total_amount_due")
                or summary.get("order_total")
                or summary.get("total_amount_due")
                or summary.get("total_commitment")
            )
            # Derive display fields from the extraction payload using the same
            # fallbacks we use elsewhere in the app.
            vendor_name = (
                data.get("vendor_name")
                or data.get("from")
                or data.get("seller")
            )
            invoice_number = (
                data.get("invoice_number")
                or data.get("po_number")
                or data.get("order_number")
            )

            # Attempt to parse an invoice date from common fields.
            raw_date = (
                data.get("invoice_date")
                or data.get("date_of_issue")
                or data.get("order_date")
                or data.get("date")
            )
            invoice_date = None
            if isinstance(raw_date, str) and raw_date.strip():
                for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
                    try:
                        invoice_date = datetime.strptime(raw_date.strip(), fmt)
                        break
                    except ValueError:
                        continue

            invoice.extraction_id = extraction_id
            invoice.vendor_name = vendor_name
            invoice.invoice_number = invoice_number
            invoice.total_amount = total
            invoice.currency = data.get("currency")
            invoice.invoice_date = invoice_date
            invoice.raw_data = data

            # Logical duplicate detection based on business keys once we have a payload.
            if (
                status == "completed"
                and invoice.user_id is not None
                and vendor_name
                and invoice_number
                and invoice_date is not None
                and total is not None
            ):
                duplicate = (
                    db_session.query(AiInvoice)
                    .filter(
                        AiInvoice.user_id == invoice.user_id,
                        AiInvoice.id != invoice.id,
                        AiInvoice.vendor_name == vendor_name,
                        AiInvoice.invoice_number == invoice_number,
                        AiInvoice.invoice_date == invoice_date,
                        AiInvoice.total_amount == total,
                        AiInvoice.status != "error",
                        AiInvoice.status != "duplicate",
                    )
                    .first()
                )
                if duplicate:
                    # We already have an invoice with the same business keys for this user.
                    # Remove this duplicate entry so it does not appear in tables or metrics.
                    db_session.delete(invoice)
                    db_session.commit()
                    return

        db_session.commit()
    finally:
        db_session.close()


def _process_ai_invoice_job(
    ai_invoice_id: int,
    file_path: str,
    content_type: str,
    provider: str,
    model_name: str,
    document_type: str,
    use_hybrid: bool,
) -> None:
    """Background task: run extraction for a stored file and update AiInvoice."""
    import os

    _update_ai_invoice_status(ai_invoice_id, "in_progress")
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        result = _perform_extraction_from_bytes(
            content=content,
            content_type=content_type,
            provider=provider,
            model_name=model_name,
            document_type=document_type,
            use_hybrid=use_hybrid,
        )
        if result["error"] or not result.get("extraction_id"):
            _update_ai_invoice_status(
                ai_invoice_id,
                "error",
                extraction_payload=None,
                extraction_id=None,
                error=str(result["error"] or "Extraction failed"),
            )
        else:
            _update_ai_invoice_status(
                ai_invoice_id,
                "completed",
                extraction_payload=result["data"],
                extraction_id=result["extraction_id"],
                error=None,
            )
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)


@router.post("/invoices/upload-batch")
async def upload_invoices_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    provider: str = Form("Gemini"),
    model_name: str = Form(""),
    document_type: str = Form("Sales Invoice bill"),
    use_hybrid: bool = Form(True),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Upload multiple invoices and enqueue background extraction jobs.

    For each file we:
    - Store it temporarily on disk
    - Create an AiInvoice row with status \"pending\"
    - Schedule a background job to run the extraction and update the row
    """
    import os

    if not files:
        raise HTTPException(status_code=400, detail="No files")

    created: List[Dict[str, Any]] = []
    db_session = SessionLocal()
    try:
        for f in files:
            if not f.filename:
                continue

            # Read content once so we can both hash it and persist it for the worker.
            content = await f.read()
            if not content:
                continue

            # Strong file-level hash used for exact duplicate detection.
            file_hash = hashlib.sha256(content).hexdigest()

            # Check for an existing non-error invoice for this user and file hash.
            existing = (
                db_session.query(AiInvoice)
                .filter(
                    AiInvoice.user_id == user.id,
                    AiInvoice.file_hash == file_hash,
                    AiInvoice.status != "error",
                )
                .first()
            )
            if existing:
                # Treat as duplicate: don't enqueue a new job, just surface the existing invoice.
                created.append(
                    {
                        "id": existing.id,
                        "file_name": f.filename,
                        "status": "duplicate",
                    }
                )
                continue

            # Persist file so the background worker can read it later.
            suffix = ".pdf" if (f.content_type or "").lower() == "application/pdf" else ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                file_path = tmp.name

            invoice = AiInvoice(
                user_id=user.id,
                extraction_id=None,
                vendor_name=None,
                invoice_number=None,
                document_type=document_type,
                invoice_date=None,
                total_amount=None,
                currency=None,
                raw_data=None,
                status="pending",
                file_hash=file_hash,
            )
            db_session.add(invoice)
            db_session.commit()
            db_session.refresh(invoice)

            background_tasks.add_task(
                _process_ai_invoice_job,
                ai_invoice_id=invoice.id,
                file_path=file_path,
                content_type=f.content_type or "",
                provider=provider,
                model_name=model_name,
                document_type=document_type,
                use_hybrid=use_hybrid,
            )

            created.append(
                {
                    "id": invoice.id,
                    "file_name": f.filename,
                    "status": invoice.status,
                }
            )
    finally:
        db_session.close()

    return {"invoices": created}


@router.get("/invoices")
def list_invoices(
    limit: int = 50,
    page: int = 1,
    search: Optional[str] = None,
    status: Optional[str] = None,
    document_type: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    List AI invoices with their background-processing status.

    Query params:
    - search: case-insensitive search on vendor_name or invoice_number
    - status: filter by status (pending, in_progress, completed, error)
    - document_type: filter by exact document/bill type
    - date_from / date_to: filter by invoice_date range (ISO date)
    - min_amount / max_amount: filter by total_amount range
    """
    from sqlalchemy import or_, func

    db_session = SessionLocal()
    try:
        if page < 1:
            page = 1

        q = db_session.query(AiInvoice).filter(AiInvoice.user_id == user.id)
        if status:
            q = q.filter(AiInvoice.status == status)
        else:
            # By default, hide invoices marked as logical duplicates from listings and counts.
            q = q.filter(AiInvoice.status != "duplicate")
        if document_type:
            q = q.filter(AiInvoice.document_type == document_type)
        if date_from:
            q = q.filter(AiInvoice.invoice_date >= date_from)
        if date_to:
            q = q.filter(AiInvoice.invoice_date <= date_to)
        if min_amount is not None:
            q = q.filter(AiInvoice.total_amount >= min_amount)
        if max_amount is not None:
            q = q.filter(AiInvoice.total_amount <= max_amount)
        if search:
            term = f"%{search.lower()}%"
            q = q.filter(
                or_(
                    func.lower(AiInvoice.vendor_name).like(term),
                    func.lower(AiInvoice.invoice_number).like(term),
                )
            )

        total = q.count()

        q = q.order_by(AiInvoice.created_at.desc())
        rows = (
            q.offset((page - 1) * limit)
            .limit(min(limit, 100))
            .all()
        )

        out: List[Dict[str, Any]] = []
        for row in rows:
            raw = row.raw_data or {}
            vendor_name = (
                row.vendor_name
                or raw.get("vendor_name")
                or raw.get("from")
                or raw.get("seller")
            )
            invoice_number = (
                row.invoice_number
                or raw.get("invoice_number")
                or raw.get("po_number")
                or raw.get("order_number")
            )
            out.append(
                {
                    "id": row.id,
                    "extraction_id": row.extraction_id,
                    "vendor_name": vendor_name,
                    "invoice_number": invoice_number,
                    "document_type": row.document_type,
                    "invoice_date": row.invoice_date.isoformat() if row.invoice_date else None,
                    "total_amount": row.total_amount,
                    "currency": row.currency,
                    "status": row.status,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                }
            )

        return {
            "items": out,
            "total": total,
            "page": page,
            "limit": limit,
        }
    finally:
        db_session.close()


@router.delete("/invoices/{invoice_id}")
def delete_invoice(
    invoice_id: int,
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Delete an AiInvoice row.

    This only removes the AI invoice record from the queue/table, it does not delete
    the underlying Extraction so learning data is preserved.
    """
    db_session = SessionLocal()
    try:
        invoice = (
            db_session.query(AiInvoice)
            .filter(AiInvoice.id == invoice_id, AiInvoice.user_id == user.id)
            .first()
        )
        if not invoice:
            raise HTTPException(status_code=404, detail="Invoice not found")
        db_session.delete(invoice)
        db_session.commit()
        return {"success": True}
    finally:
        db_session.close()
