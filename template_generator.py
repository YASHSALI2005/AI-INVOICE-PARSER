"""
Auto-Template Generator - Creates invoice2data templates from LLM extractions.

When an LLM successfully extracts an invoice, this module can:
1. Generate keywords for vendor matching
2. Create regex patterns for extracted fields
3. Output a valid invoice2data YAML template
"""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime


def escape_regex(text: str) -> str:
    """Escape special regex characters."""
    special_chars = r'\.^$*+?{}[]|()'
    for char in special_chars:
        text = text.replace(char, '\\' + char)
    return text


def generate_amount_pattern(amount: float, currency: str = "") -> str:
    """Generate regex pattern for amounts."""
    # Handle different amount formats
    amount_str = f"{amount:.2f}"
    whole, decimal = amount_str.split(".")
    
    # Pattern that matches: $4.11, 4.11, $4,11, etc.
    patterns = [
        f"[$€£]?\\s*{whole}[.,]{decimal}",
        f"{whole}[.,]{decimal}\\s*[$€£{currency}]?",
    ]
    return f"({patterns[0]}|{patterns[1]})"


def generate_date_pattern(date_str: str) -> str:
    """Generate regex pattern for dates."""
    # Common date formats
    patterns = [
        r"(\d{4}-\d{2}-\d{2})",  # YYYY-MM-DD
        r"(\d{2}/\d{2}/\d{4})",  # DD/MM/YYYY or MM/DD/YYYY
        r"(\d{2}-\d{2}-\d{4})",  # DD-MM-YYYY
        r"([A-Za-z]+\s+\d{1,2},?\s+\d{4})",  # Month DD, YYYY
        r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})",  # DD Month YYYY
    ]
    return "|".join(patterns)


def generate_invoice_number_pattern(invoice_num: str) -> str:
    """Generate regex pattern for invoice numbers."""
    if not invoice_num:
        return r"(\w+[-/]?\w+)"
    
    # Analyze the invoice number format
    has_letters = bool(re.search(r'[A-Za-z]', invoice_num))
    has_numbers = bool(re.search(r'\d', invoice_num))
    has_separator = bool(re.search(r'[-/]', invoice_num))
    
    if has_separator:
        # Pattern with separators like INV/2023/001
        return r"([\w]+[-/][\w]+(?:[-/][\w]+)*)"
    elif has_letters and has_numbers:
        # Alphanumeric like INV001
        return r"([A-Za-z]+\d+|\d+[A-Za-z]+)"
    else:
        # Numeric only
        return r"(\d+)"


def extract_keywords(vendor_name: str, extracted_data: Dict[str, Any]) -> List[str]:
    """Extract keywords for template matching."""
    keywords = []
    
    # Vendor name parts
    if vendor_name:
        # Split vendor name into significant words
        words = vendor_name.split()
        for word in words:
            # Skip common words
            if word.lower() not in ["inc", "inc.", "llc", "ltd", "limited", "corporation", "corp", "corp.", "the", "and", "&"]:
                if len(word) > 2:
                    keywords.append(word)
    
    # Add account number if present
    summary = extracted_data.get("summary", {})
    if summary.get("account_number"):
        keywords.append(str(summary["account_number"]))
    
    return keywords[:5]  # Limit to 5 keywords


def generate_template(
    vendor_name: str,
    extracted_data: Dict[str, Any],
    confidence_scores: Optional[Dict[str, float]] = None,
) -> str:
    """
    Generate an invoice2data-style YAML template from extracted data.
    
    Args:
        vendor_name: Name of the vendor
        extracted_data: The LLM-extracted data
        confidence_scores: Optional confidence scores per field
        
    Returns:
        YAML template string
    """
    keywords = extract_keywords(vendor_name, extracted_data)
    
    # Build template
    lines = [
        f"# Auto-generated template for {vendor_name}",
        f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"# This template was created from LLM extraction",
        "",
        f"issuer: {vendor_name}",
        "",
        "keywords:",
    ]
    
    for kw in keywords:
        lines.append(f"  - {kw}")
    
    lines.extend([
        "",
        "fields:",
    ])
    
    # Amount field
    if extracted_data.get("total_amount") is not None:
        amount = extracted_data["total_amount"]
        currency = extracted_data.get("currency", "")
        lines.extend([
            "  amount:",
            "    parser: regex",
            f"    regex: {generate_amount_pattern(amount, currency)}",
            "    type: float",
        ])
    
    # Date field
    if extracted_data.get("date"):
        lines.extend([
            "  date:",
            "    parser: regex",
            f"    regex: {generate_date_pattern(extracted_data['date'])}",
            "    type: date",
        ])
    
    # Invoice number field
    if extracted_data.get("invoice_number"):
        lines.extend([
            "  invoice_number:",
            "    parser: regex",
            f"    regex: Invoice\\s*#?:?\\s*{generate_invoice_number_pattern(extracted_data['invoice_number'])}",
        ])
    
    # Partner/Vendor name (static)
    lines.extend([
        "  partner_name:",
        "    parser: static",
        f"    value: \"{vendor_name}\"",
    ])
    
    # Summary fields
    summary = extracted_data.get("summary", {})
    
    if summary.get("tax") is not None:
        lines.extend([
            "  amount_tax:",
            "    parser: regex",
            f"    regex: Tax\\s*:?\\s*{generate_amount_pattern(summary['tax'])}",
            "    type: float",
        ])
    
    if summary.get("subtotal") is not None:
        lines.extend([
            "  amount_untaxed:",
            "    parser: regex",
            f"    regex: Subtotal\\s*:?\\s*{generate_amount_pattern(summary['subtotal'])}",
            "    type: float",
        ])
    
    # Options
    lines.extend([
        "",
        "options:",
        f"  currency: {extracted_data.get('currency', 'USD')}",
        "  date_formats:",
        "    - '%Y-%m-%d'",
        "    - '%d/%m/%Y'",
        "    - '%m/%d/%Y'",
        "    - '%B %d, %Y'",
    ])
    
    # Line items (if present)
    line_items = extracted_data.get("line_items", [])
    if line_items and len(line_items) > 0:
        lines.extend([
            "",
            "# Note: Line items pattern needs manual refinement",
            "# lines:",
            "#   start: \"Description\"",
            "#   end: \"Total\"",
            "#   first_line: (?P<description>.+)\\s+(?P<total>\\d+\\.\\d+)",
        ])
    
    return "\n".join(lines)


def should_generate_template(
    extracted_data: Dict[str, Any],
    confidence: float,
    min_confidence: float = 0.8,
    min_fields: int = 4,
) -> bool:
    """
    Determine if we should generate a template from this extraction.
    
    Criteria:
    - High confidence extraction
    - Has minimum required fields
    - Vendor name is present
    """
    if confidence < min_confidence:
        return False
    
    # Check required fields
    required_fields = ["invoice_number", "date", "total_amount", "vendor_name"]
    present_fields = sum(1 for f in required_fields if extracted_data.get(f) is not None)
    
    if present_fields < min_fields:
        return False
    
    if not extracted_data.get("vendor_name"):
        return False
    
    return True


def improve_template_with_corrections(
    existing_template: str,
    corrections: List[Dict[str, Any]],
) -> str:
    """
    Improve an existing template based on user corrections.
    
    This analyzes corrections to find patterns and update regex.
    """
    # For now, just add a comment about corrections
    # A more sophisticated version would analyze correction patterns
    
    if not corrections:
        return existing_template
    
    correction_summary = f"\n# Template has {len(corrections)} user corrections applied\n"
    correction_summary += f"# Last correction: {corrections[-1].get('timestamp', 'unknown')}\n"
    
    # Insert after the header
    lines = existing_template.split("\n")
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith("issuer:"):
            header_end = i
            break
    
    lines.insert(header_end, correction_summary)
    
    return "\n".join(lines)
