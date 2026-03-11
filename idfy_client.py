"""
IDfy verification client.

Two-phase flow:
  1. create_*_task()  — fire-and-forget during extraction; returns a request_id.
  2. resolve_task()   — called later (e.g. on page load) to fetch the result.
"""

import time
import uuid
from typing import Literal, Optional, Tuple

import requests

from config import get_settings

VerificationStatus = Literal["valid", "invalid", "unknown", "error", "pending"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _can_call_idfy() -> bool:
    s = get_settings()
    return bool(s.idfy_account_id and s.idfy_api_key)


def _headers() -> dict:
    s = get_settings()
    return {
        "Content-Type": "application/json",
        "account-id": s.idfy_account_id,
        "api-key": s.idfy_api_key,
    }


def _post_task(url: str, payload: dict, label: str) -> Optional[str]:
    """POST to create an async IDfy task. Returns the request_id or None."""
    print(f"[IDfy] Creating {label} task...")
    try:
        resp = requests.post(url, json=payload, headers=_headers(), timeout=30)
    except Exception as e:
        print(f"[IDfy] {label} task creation failed: {e}")
        return None

    print(f"[IDfy] {label} response: status={resp.status_code} body={resp.text[:500]}")

    if not (200 <= resp.status_code < 300):
        return None

    try:
        body = resp.json() or {}
    except Exception:
        return None

    request_id = body.get("request_id")
    if isinstance(request_id, str) and request_id:
        print(f"[IDfy] {label} got request_id={request_id}")
        return request_id

    print(f"[IDfy] {label} no request_id in response: {body}")
    return None


def _interpret_task(task: dict, label: str) -> VerificationStatus:
    """Given a polled IDfy task dict, decide the verification status."""
    task_status = str(task.get("status", "") or "").lower()

    if task_status in ("failed", "error", "errored", "cancelled"):
        print(f"[IDfy] {label} task failed (status={task_status})")
        return "invalid"

    result = task.get("result") or {}
    print(f"[IDfy] {label} raw result keys: {list(result.keys())}")

    # Bank-specific: account_exists is the most explicit signal.
    acct_exists = str(result.get("account_exists", "")).strip().upper()
    if acct_exists:
        print(f"[IDfy] {label} account_exists = '{acct_exists}'")
        if acct_exists == "YES":
            return "valid"
        if acct_exists == "NO":
            return "invalid"

    # Check nested source_output / extraction_output first.
    for key in ("source_output", "extraction_output"):
        section = result.get(key) or {}
        raw = section.get("status")
        if raw is not None:
            s = str(raw).strip().lower()
            print(f"[IDfy] {label} {key}.status = '{s}'")
            if "id_found" in s and "not_found" not in s:
                return "valid"
            if "id_not_found" in s:
                return "invalid"

    # Direct result.status (bank accounts, some other task types).
    raw = result.get("status")
    if raw is not None:
        s = str(raw).strip().lower()
        print(f"[IDfy] {label} result.status = '{s}'")
        if "id_found" in s and "not_found" not in s:
            return "valid"
        if "id_not_found" in s:
            return "invalid"

    # Fallback: source_output.account_status
    source_output = result.get("source_output") or {}
    acct = str(source_output.get("account_status", "")).strip().lower()
    if acct:
        print(f"[IDfy] {label} account_status = '{acct}'")
        if acct in ("active", "verified", "valid"):
            return "valid"
        if acct in ("inactive", "invalid", "not_found", "closed"):
            return "invalid"

    print(f"[IDfy] {label} could not determine status: {str(result)[:300]}")
    return "unknown"


# ---------------------------------------------------------------------------
# Phase 1 — Task creation (used during extraction)
# ---------------------------------------------------------------------------

def create_gstin_task(gstin: str) -> Optional[str]:
    s = get_settings()
    if not _can_call_idfy() or not s.idfy_gstin_url or not gstin:
        return None
    payload = {
        "task_id": str(uuid.uuid4()),
        "group_id": str(uuid.uuid4()),
        "data": {"gstin": gstin, "filing_details": True, "e_invoice_details": True},
    }
    return _post_task(s.idfy_gstin_url, payload, f"GSTIN({gstin})")


def create_msme_task(msme_number: str) -> Optional[str]:
    s = get_settings()
    if not _can_call_idfy() or not s.idfy_msme_url or not msme_number:
        return None
    payload = {
        "task_id": str(uuid.uuid4()),
        "group_id": str(uuid.uuid4()),
        "data": {"udyam_number": msme_number},
    }
    return _post_task(s.idfy_msme_url, payload, f"MSME({msme_number})")


def create_bank_task(account_number: str, ifsc: str) -> Optional[str]:
    s = get_settings()
    if not _can_call_idfy() or not s.idfy_bank_ifsc_url or not account_number or not ifsc:
        return None
    payload = {
        "task_id": str(uuid.uuid4()),
        "group_id": str(uuid.uuid4()),
        "data": {
            # Match IDfy validate_bank_account API field names
            "bank_account_no": account_number,
            "bank_ifsc_code": ifsc,
            "nf_verification": True,
        },
    }
    return _post_task(s.idfy_bank_ifsc_url, payload, f"Bank({account_number}/{ifsc})")


# ---------------------------------------------------------------------------
# Phase 2 — Resolve a single request_id (used on page load)
# ---------------------------------------------------------------------------

RESOLVE_POLL_INTERVAL = 3   # seconds between polls
RESOLVE_MAX_ATTEMPTS = 4    # bank tasks take ~4s, so give up to ~12s


def resolve_task_with_raw(request_id: str, label: str = "") -> Tuple[VerificationStatus, dict]:
    """
    Fetch the task dict for a previously created IDfy task and interpret it.
    Returns (status, raw_task_dict).
    """
    s = get_settings()
    if not request_id or not s.idfy_tasks_url:
        return "unknown", {}

    for attempt in range(RESOLVE_MAX_ATTEMPTS):
        if attempt > 0:
            time.sleep(RESOLVE_POLL_INTERVAL)

        try:
            resp = requests.get(
                f"{s.idfy_tasks_url}?request_id={request_id}",
                headers=_headers(),
                timeout=30,
            )
            print(f"[IDfy] Response for {label} attempt {attempt + 1}: {resp.text}")
        except Exception as e:
            print(f"[IDfy] Resolve {label} attempt {attempt + 1} network error: {e}")
            continue

        if not (200 <= resp.status_code < 300):
            print(f"[IDfy] Resolve {label} attempt {attempt + 1} status={resp.status_code}")
            continue

        try:
            body = resp.json()
        except Exception:
            continue

        if not isinstance(body, list) or not body:
            continue

        first = body[0] or {}
        task_status = str(first.get("status", "") or "").lower()

        if task_status in ("completed", "failed", "error", "errored", "cancelled"):
            status = _interpret_task(first, label or request_id)
            return status, first

        print(f"[IDfy] Resolve {label} attempt {attempt + 1} still {task_status}")

    return "pending", {}


def resolve_task(request_id: str, label: str = "") -> VerificationStatus:
    """
    Backwards-compatible wrapper that only returns the interpreted status.
    """
    status, _ = resolve_task_with_raw(request_id, label)
    return status


# ---------------------------------------------------------------------------
# PAN helpers — verify via button (needs name + DOB)
# ---------------------------------------------------------------------------

def create_pan_task(pan: str, full_name: str, dob: str) -> Optional[str]:
    """Create a PAN verification task and return request_id."""
    s = get_settings()
    if not _can_call_idfy() or not s.idfy_pan_url or not s.idfy_tasks_url or not pan:
        return None

    payload = {
        "task_id": str(uuid.uuid4()),
        "group_id": str(uuid.uuid4()),
        "data": {"id_number": pan, "full_name": full_name, "dob": dob},
    }
    return _post_task(s.idfy_pan_url, payload, f"PAN({pan})")


def verify_pan(pan: str, full_name: str, dob: str) -> Tuple[VerificationStatus, Optional[str]]:
    """
    Validate PAN via IDfy (requires full name + date of birth).
    Returns (status, request_id) so callers can persist the request_id.
    """
    request_id = create_pan_task(pan, full_name, dob)
    if not request_id:
        return "error", None

    # PAN is on-demand so we wait a bit longer before giving up.
    status, _ = resolve_task_with_raw(request_id, f"PAN({pan})")
    if status == "pending":
        # If still pending after our quick check, let the caller re-resolve later.
        status = "pending"
    return status, request_id
