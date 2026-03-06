from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.deps import get_db, get_current_user
from db.models import AiInvoice, InvoiceAudit, InvoiceAuditItem, User

router = APIRouter(prefix="/api", tags=["invoice-audits"])


class CreateInvoiceAuditBody(BaseModel):
  final_invoice_id: int
  supporting_invoice_ids: List[int]
  name: Optional[str] = None


@router.post("/invoice-audits")
def create_invoice_audit(
  body: CreateInvoiceAuditBody,
  db: Session = Depends(get_db),
  user: User = Depends(get_current_user),
) -> Dict[str, Any]:
  """Create an invoice audit for the current user."""
  if not body.supporting_invoice_ids:
    raise HTTPException(status_code=400, detail="At least one supporting invoice is required")

  if body.final_invoice_id in body.supporting_invoice_ids:
    raise HTTPException(status_code=400, detail="Final invoice cannot also be a supporting invoice")

  # Load all referenced invoices and ensure they belong to the current user.
  invoice_ids = [body.final_invoice_id] + body.supporting_invoice_ids
  invoices = (
    db.query(AiInvoice)
    .filter(AiInvoice.id.in_(invoice_ids), AiInvoice.user_id == user.id)
    .all()
  )
  if len(invoices) != len(set(invoice_ids)):
    raise HTTPException(status_code=404, detail="One or more invoices not found for this user")

  # Simple uniqueness: avoid duplicates with same final invoice for same user and same set of supporting ids.
  existing = (
    db.query(InvoiceAudit)
    .filter(
      InvoiceAudit.user_id == user.id,
      InvoiceAudit.final_invoice_id == body.final_invoice_id,
    )
    .first()
  )

  audit = InvoiceAudit(
    user_id=user.id,
    final_invoice_id=body.final_invoice_id,
    name=body.name,
  )
  db.add(audit)
  db.flush()

  for inv_id in body.supporting_invoice_ids:
    item = InvoiceAuditItem(audit_id=audit.id, invoice_id=inv_id)
    db.add(item)

  db.commit()
  db.refresh(audit)

  return get_invoice_audit(audit.id, db=db, user=user)


@router.get("/invoice-audits")
def list_invoice_audits(
  limit: int = 50,
  page: int = 1,
  db: Session = Depends(get_db),
  user: User = Depends(get_current_user),
) -> Dict[str, Any]:
  """List saved invoice audits for the current user."""
  if page < 1:
    page = 1

  q = db.query(InvoiceAudit).filter(InvoiceAudit.user_id == user.id)
  total = q.count()

  audits = (
    q.order_by(InvoiceAudit.created_at.desc())
    .offset((page - 1) * limit)
    .limit(limit)
    .all()
  )

  # Fetch final invoices in batch
  final_ids = [a.final_invoice_id for a in audits]
  final_invoices: Dict[int, AiInvoice] = {}
  if final_ids:
    rows = (
      db.query(AiInvoice)
      .filter(AiInvoice.id.in_(final_ids), AiInvoice.user_id == user.id)
      .all()
    )
    final_invoices = {row.id: row for row in rows}

  items: List[Dict[str, Any]] = []
  for audit in audits:
    final_inv = final_invoices.get(audit.final_invoice_id)
    items.append(
      {
        "id": audit.id,
        "name": audit.name,
        "created_at": audit.created_at.isoformat() if audit.created_at else None,
        "updated_at": audit.updated_at.isoformat() if audit.updated_at else None,
        "final_invoice_id": audit.final_invoice_id,
        "final_invoice_number": getattr(final_inv, "invoice_number", None),
        "final_invoice_vendor": getattr(final_inv, "vendor_name", None),
        "final_total_amount": getattr(final_inv, "total_amount", None),
      }
    )

  return {
    "items": items,
    "total": total,
    "page": page,
    "limit": limit,
  }


@router.get("/invoice-audits/{audit_id}")
def get_invoice_audit(
  audit_id: int,
  db: Session = Depends(get_db),
  user: User = Depends(get_current_user),
) -> Dict[str, Any]:
  """Get a single invoice audit with hydrated invoice data and computed totals."""
  audit = (
    db.query(InvoiceAudit)
    .filter(InvoiceAudit.id == audit_id, InvoiceAudit.user_id == user.id)
    .first()
  )
  if not audit:
    raise HTTPException(status_code=404, detail="Audit not found")

  # Final invoice
  final_invoice = (
    db.query(AiInvoice)
    .filter(
      AiInvoice.id == audit.final_invoice_id,
      AiInvoice.user_id == user.id,
    )
    .first()
  )
  if not final_invoice:
    raise HTTPException(status_code=404, detail="Final invoice not found")

  # Supporting invoices
  item_rows = (
    db.query(InvoiceAuditItem)
    .filter(InvoiceAuditItem.audit_id == audit.id)
    .all()
  )
  supporting_ids = [item.invoice_id for item in item_rows]
  supporting_invoices: List[AiInvoice] = []
  if supporting_ids:
    supporting_invoices = (
      db.query(AiInvoice)
      .filter(AiInvoice.id.in_(supporting_ids), AiInvoice.user_id == user.id)
      .all()
    )

  def to_record(inv: AiInvoice) -> Dict[str, Any]:
    return {
      "id": inv.id,
      "extraction_id": inv.extraction_id,
      "vendor_name": inv.vendor_name,
      "invoice_number": inv.invoice_number,
      "document_type": inv.document_type,
      "invoice_date": inv.invoice_date.isoformat() if inv.invoice_date else None,
      "total_amount": inv.total_amount,
      "currency": inv.currency,
      "status": inv.status,
      "created_at": inv.created_at.isoformat() if inv.created_at else None,
      "updated_at": inv.updated_at.isoformat() if inv.updated_at else None,
    }

  final_total = final_invoice.total_amount or 0
  supporting_total = sum((inv.total_amount or 0) for inv in supporting_invoices)
  difference = supporting_total - final_total

  return {
    "id": audit.id,
    "name": audit.name,
    "created_at": audit.created_at.isoformat() if audit.created_at else None,
    "updated_at": audit.updated_at.isoformat() if audit.updated_at else None,
    "final_invoice": to_record(final_invoice),
    "supporting_invoices": [to_record(inv) for inv in supporting_invoices],
    "final_total": final_total,
    "supporting_total": supporting_total,
    "difference": difference,
  }

