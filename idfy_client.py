import time
import uuid
from typing import Literal

import requests

from config import get_settings

VerificationStatus = Literal["valid", "invalid", "unknown", "error"]

POLL_INTERVAL = 3      # seconds between polls
POLL_MAX_ATTEMPTS = 5  # try up to 5 times (≈15s max wait)


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


def _poll_task_result(request_id: str) -> dict:
    """
    Poll the IDfy tasks endpoint until the task completes or we exhaust retries.
    Returns the first task dict from the response, or empty dict on failure.
    """
    s = get_settings()
    for attempt in range(POLL_MAX_ATTEMPTS):
        time.sleep(POLL_INTERVAL)
        try:
            resp = requests.get(
                f"{s.idfy_tasks_url}?request_id={request_id}",
                headers=_headers(),
                timeout=30,
            )
        except Exception as e:
            print(f"[IDfy] Poll attempt {attempt + 1} network error: {e}")
            continue

        if not (200 <= resp.status_code < 300):
            print(f"[IDfy] Poll attempt {attempt + 1} status={resp.status_code} body={resp.text[:300]}")
            continue

        try:
            body = resp.json()
        except Exception:
            print(f"[IDfy] Poll attempt {attempt + 1} JSON parse error")
            continue

        if not isinstance(body, list) or not body:
            print(f"[IDfy] Poll attempt {attempt + 1} empty or non-list body: {str(body)[:300]}")
            continue

        first = body[0] or {}
        task_status = first.get("status", "")

        if task_status == "completed":
            return first

        print(f"[IDfy] Poll attempt {attempt + 1} task not ready yet (status={task_status})")

    print(f"[IDfy] Gave up polling after {POLL_MAX_ATTEMPTS} attempts for request_id={request_id}")
    return {}


def _extract_source_status(task: dict) -> str:
    """Extract the source_output.status string from a completed task dict."""
    result = task.get("result") or {}
    source_output = result.get("source_output") or {}
    raw = source_output.get("status")
    if raw is not None:
        return str(raw).strip().lower()

    extraction_output = result.get("extraction_output") or {}
    raw = extraction_output.get("status")
    if raw is not None:
        return str(raw).strip().lower()

    raw = result.get("status")
    if raw is not None:
        return str(raw).strip().lower()

    return ""


def verify_gstin(gstin: str) -> VerificationStatus:
    """
    Validate GSTIN via IDfy async task API.

    Step 1: POST to create an async verification task.
    Step 2: Poll GET until the task completes.
    Step 3: Read source_output.status → 'id_found' or 'id_not_found'.
    """
    s = get_settings()
    if not _can_call_idfy() or not s.idfy_gstin_url or not s.idfy_tasks_url or not gstin:
        print(f"[IDfy] GSTIN check skipped: can_call={_can_call_idfy()} gstin_url={bool(s.idfy_gstin_url)} tasks_url={bool(s.idfy_tasks_url)} gstin={bool(gstin)}")
        return "unknown"

    task_id = str(uuid.uuid4())
    group_id = str(uuid.uuid4())
    payload = {
        "task_id": task_id,
        "group_id": group_id,
        "data": {
            "gstin": gstin,
            "filing_details": True,
            "e_invoice_details": True,
        },
    }

    print(f"[IDfy] Creating GSTIN task for {gstin}...")
    try:
        resp = requests.post(s.idfy_gstin_url, json=payload, headers=_headers(), timeout=30)
    except Exception as e:
        print(f"[IDfy] GSTIN task creation failed: {e}")
        return "error"

    print(f"[IDfy] Task creation response: status={resp.status_code} body={resp.text[:500]}")

    if not (200 <= resp.status_code < 300):
        if 400 <= resp.status_code < 500:
            return "invalid"
        return "error"

    try:
        body = resp.json() or {}
    except Exception:
        print("[IDfy] Could not parse task creation JSON")
        return "error"

    request_id = body.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        print(f"[IDfy] No request_id in response: {body}")
        return "error"

    print(f"[IDfy] Got request_id={request_id}, polling for result...")

    task = _poll_task_result(request_id)
    if not task:
        return "error"

    status_str = _extract_source_status(task)
    print(f"[IDfy] GSTIN source_output.status = '{status_str}'")

    if "id_found" in status_str and "not_found" not in status_str:
        return "valid"
    if "id_not_found" in status_str:
        return "invalid"
    return "unknown"


def verify_msme(msme_number: str) -> VerificationStatus:
    """Validate MSME registration number via IDfy async task API."""
    s = get_settings()
    if not _can_call_idfy() or not s.idfy_msme_url or not s.idfy_tasks_url or not msme_number:
        return "unknown"

    task_id = str(uuid.uuid4())
    group_id = str(uuid.uuid4())
    payload = {
        "task_id": task_id,
        "group_id": group_id,
        "data": {"msme_number": msme_number},
    }

    print(f"[IDfy] Creating MSME task for {msme_number}...")
    try:
        resp = requests.post(s.idfy_msme_url, json=payload, headers=_headers(), timeout=30)
    except Exception as e:
        print(f"[IDfy] MSME task creation failed: {e}")
        return "error"

    print(f"[IDfy] MSME task response: status={resp.status_code} body={resp.text[:500]}")

    if not (200 <= resp.status_code < 300):
        if 400 <= resp.status_code < 500:
            return "invalid"
        return "error"

    try:
        body = resp.json() or {}
    except Exception:
        return "error"

    request_id = body.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        return "error"

    task = _poll_task_result(request_id)
    if not task:
        return "error"

    status_str = _extract_source_status(task)
    print(f"[IDfy] MSME source_output.status = '{status_str}'")

    if "id_found" in status_str and "not_found" not in status_str:
        return "valid"
    if "id_not_found" in status_str:
        return "invalid"
    return "unknown"


def verify_bank_account(account_number: str, ifsc: str) -> VerificationStatus:
    """Validate bank account + IFSC via IDfy async task API."""
    s = get_settings()
    if not _can_call_idfy() or not s.idfy_bank_ifsc_url or not s.idfy_tasks_url or not account_number or not ifsc:
        return "unknown"

    task_id = str(uuid.uuid4())
    group_id = str(uuid.uuid4())
    payload = {
        "task_id": task_id,
        "group_id": group_id,
        "data": {"account_number": account_number, "ifsc": ifsc},
    }

    print(f"[IDfy] Creating bank account task for {account_number} / {ifsc}...")
    try:
        resp = requests.post(s.idfy_bank_ifsc_url, json=payload, headers=_headers(), timeout=30)
    except Exception as e:
        print(f"[IDfy] Bank task creation failed: {e}")
        return "error"

    print(f"[IDfy] Bank task response: status={resp.status_code} body={resp.text[:500]}")

    if not (200 <= resp.status_code < 300):
        if 400 <= resp.status_code < 500:
            return "invalid"
        return "error"

    try:
        body = resp.json() or {}
    except Exception:
        return "error"

    request_id = body.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        return "error"

    task = _poll_task_result(request_id)
    if not task:
        return "error"

    status_str = _extract_source_status(task)
    print(f"[IDfy] Bank source_output.status = '{status_str}'")

    if "id_found" in status_str and "not_found" not in status_str:
        return "valid"
    if "id_not_found" in status_str:
        return "invalid"
    return "unknown"
