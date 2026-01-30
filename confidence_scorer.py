"""
Confidence Scorer - Validates extractions and flags uncertain fields.

Provides:
- Field-level confidence scores
- Overall extraction confidence
- Validation checks (format, consistency)
- Flags for human review
"""

import re
from typing import Dict, Any, List, Tuple
from datetime import datetime


def validate_date_format(date_str: str) -> Tuple[bool, float]:
    """
    Validate date format and return (is_valid, confidence).
    """
    if not date_str:
        return False, 0.0
    
    # Try common formats
    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
    ]
    
    for fmt in formats:
        try:
            parsed = datetime.strptime(date_str, fmt)
            # Check if date is reasonable (not too far in past/future)
            now = datetime.now()
            years_diff = abs((now - parsed).days / 365)
            if years_diff > 20:
                return True, 0.6  # Valid format but suspicious date
            return True, 1.0
        except ValueError:
            continue
    
    # Check if it looks like a date
    if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
        return True, 0.9
    if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', date_str):
        return True, 0.8
    
    return False, 0.3


def validate_amount(amount: Any) -> Tuple[bool, float]:
    """
    Validate amount and return (is_valid, confidence).
    """
    if amount is None:
        return False, 0.0
    
    try:
        num = float(amount)
        if num < 0:
            return True, 0.5  # Negative amounts are suspicious
        if num == 0:
            return True, 0.7  # Zero might be valid but unusual
        if num > 10000000:
            return True, 0.6  # Very large amounts need review
        return True, 1.0
    except (ValueError, TypeError):
        return False, 0.2


def validate_currency(currency: str) -> Tuple[bool, float]:
    """
    Validate currency code and return (is_valid, confidence).
    """
    if not currency:
        return False, 0.0
    
    # Common ISO 4217 codes
    valid_codes = {
        "USD", "EUR", "GBP", "JPY", "CNY", "INR", "AUD", "CAD", "CHF",
        "HKD", "SGD", "SEK", "KRW", "NOK", "NZD", "MXN", "TWD", "ZAR",
        "BRL", "DKK", "PLN", "THB", "IDR", "MYR", "PHP", "CZK", "ILS",
        "CLP", "PKR", "EGP", "COP", "SAR", "AED", "RON", "VND", "HUF",
    }
    
    currency_upper = currency.upper().strip()
    
    if currency_upper in valid_codes:
        return True, 1.0
    
    # Check if it's a symbol
    if currency in ["$", "€", "£", "¥", "₹", "₩"]:
        return True, 0.9
    
    # Check if it looks like a currency code
    if re.match(r'^[A-Z]{3}$', currency_upper):
        return True, 0.7
    
    return False, 0.3


def validate_invoice_number(invoice_num: str) -> Tuple[bool, float]:
    """
    Validate invoice number and return (is_valid, confidence).
    """
    if not invoice_num:
        return False, 0.0
    
    invoice_num = str(invoice_num).strip()
    
    # Too short
    if len(invoice_num) < 2:
        return True, 0.5
    
    # Too long (might have captured extra text)
    if len(invoice_num) > 50:
        return True, 0.4
    
    # Contains suspicious characters
    if re.search(r'[\n\r\t]', invoice_num):
        return True, 0.3
    
    # Looks like a proper invoice number
    if re.match(r'^[A-Za-z0-9\-/]+$', invoice_num):
        return True, 1.0
    
    return True, 0.7


def validate_vendor_name(vendor_name: str) -> Tuple[bool, float]:
    """
    Validate vendor name and return (is_valid, confidence).
    """
    if not vendor_name:
        return False, 0.0
    
    vendor_name = str(vendor_name).strip()
    
    # Too short
    if len(vendor_name) < 2:
        return True, 0.4
    
    # Too long (might have captured address or extra text)
    if len(vendor_name) > 100:
        return True, 0.5
    
    # Contains newlines (probably captured too much)
    if '\n' in vendor_name:
        return True, 0.4
    
    # Looks reasonable
    return True, 1.0


def validate_line_items(line_items: List[Dict[str, Any]]) -> Tuple[bool, float]:
    """
    Validate line items and return (is_valid, confidence).
    """
    if not line_items:
        return True, 0.5  # No line items might be okay
    
    if not isinstance(line_items, list):
        return False, 0.2
    
    valid_items = 0
    total_items = len(line_items)
    
    for item in line_items:
        if not isinstance(item, dict):
            continue
        
        # Check if item has meaningful content
        has_description = bool(item.get("description"))
        has_total = item.get("total") is not None
        
        if has_description or has_total:
            valid_items += 1
    
    if total_items == 0:
        return True, 0.5
    
    confidence = valid_items / total_items
    return True, confidence


def check_amount_consistency(extracted_data: Dict[str, Any]) -> Tuple[bool, float, str]:
    """
    Check if amounts are consistent (line items sum up to total, etc.).
    Returns (is_consistent, confidence, message).
    """
    total_amount = extracted_data.get("total_amount")
    summary = extracted_data.get("summary", {})
    line_items = extracted_data.get("line_items", [])
    
    issues = []
    
    # Check if line items sum up to subtotal or total
    if line_items and total_amount:
        line_total = sum(
            item.get("total", 0) or 0 
            for item in line_items 
            if isinstance(item, dict)
        )
        
        subtotal = summary.get("subtotal") or total_amount
        
        if line_total > 0:
            diff = abs(line_total - subtotal)
            if diff > 0.01:
                # Allow for tax difference
                tax = summary.get("tax", 0) or 0
                if abs(line_total + tax - total_amount) > 0.01:
                    issues.append(f"Line items ({line_total}) don't match total ({total_amount})")
    
    # Check subtotal + tax = total
    if summary.get("subtotal") and summary.get("tax") and total_amount:
        expected_total = (summary.get("subtotal") or 0) + (summary.get("tax") or 0)
        if abs(expected_total - total_amount) > 0.01:
            issues.append(f"Subtotal + Tax ({expected_total}) != Total ({total_amount})")
    
    if issues:
        return False, 0.6, "; ".join(issues)
    
    return True, 1.0, "Amounts are consistent"


def calculate_confidence(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate confidence scores for the entire extraction.
    
    Returns:
        {
            "overall": float,
            "fields": {
                "field_name": {"score": float, "valid": bool, "message": str}
            },
            "flags": [str],  # Issues that need human review
        }
    """
    result = {
        "overall": 0.0,
        "fields": {},
        "flags": [],
    }
    
    scores = []
    weights = []
    
    # Validate each field
    # Invoice number (weight: 1.0)
    valid, score = validate_invoice_number(extracted_data.get("invoice_number"))
    result["fields"]["invoice_number"] = {"score": score, "valid": valid}
    scores.append(score)
    weights.append(1.0)
    if score < 0.7:
        result["flags"].append("Invoice number may be incorrect")
    
    # Date (weight: 1.0)
    valid, score = validate_date_format(extracted_data.get("date"))
    result["fields"]["date"] = {"score": score, "valid": valid}
    scores.append(score)
    weights.append(1.0)
    if score < 0.7:
        result["flags"].append("Date format may be incorrect")
    
    # Amount (weight: 1.5 - most important)
    valid, score = validate_amount(extracted_data.get("total_amount"))
    result["fields"]["total_amount"] = {"score": score, "valid": valid}
    scores.append(score)
    weights.append(1.5)
    if score < 0.7:
        result["flags"].append("Total amount may be incorrect")
    
    # Currency (weight: 0.5)
    valid, score = validate_currency(extracted_data.get("currency"))
    result["fields"]["currency"] = {"score": score, "valid": valid}
    scores.append(score)
    weights.append(0.5)
    if score < 0.7:
        result["flags"].append("Currency may be incorrect")
    
    # Vendor name (weight: 1.0)
    valid, score = validate_vendor_name(extracted_data.get("vendor_name"))
    result["fields"]["vendor_name"] = {"score": score, "valid": valid}
    scores.append(score)
    weights.append(1.0)
    if score < 0.7:
        result["flags"].append("Vendor name may be incorrect")
    
    # Line items (weight: 0.8)
    valid, score = validate_line_items(extracted_data.get("line_items", []))
    result["fields"]["line_items"] = {"score": score, "valid": valid}
    scores.append(score)
    weights.append(0.8)
    if score < 0.7:
        result["flags"].append("Line items may be incomplete")
    
    # Amount consistency check
    consistent, consistency_score, message = check_amount_consistency(extracted_data)
    result["fields"]["consistency"] = {"score": consistency_score, "valid": consistent, "message": message}
    scores.append(consistency_score)
    weights.append(1.0)
    if not consistent:
        result["flags"].append(message)
    
    # Calculate weighted average
    if sum(weights) > 0:
        result["overall"] = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    
    return result


def needs_human_review(confidence_result: Dict[str, Any], threshold: float = 0.75) -> bool:
    """Check if extraction needs human review based on confidence."""
    return confidence_result["overall"] < threshold or len(confidence_result["flags"]) > 0


def get_confidence_summary(confidence_result: Dict[str, Any]) -> str:
    """Get a human-readable summary of confidence scores."""
    overall = confidence_result["overall"]
    
    if overall >= 0.9:
        level = "High"
        emoji = "✅"
    elif overall >= 0.75:
        level = "Medium"
        emoji = "⚠️"
    else:
        level = "Low"
        emoji = "❌"
    
    summary = f"{emoji} Confidence: {level} ({overall:.0%})"
    
    if confidence_result["flags"]:
        summary += f"\n⚠️ {len(confidence_result['flags'])} issue(s) found"
    
    return summary
