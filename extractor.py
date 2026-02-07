import os
import json
import io
from typing import List, Dict, Union, Optional, Any

import google.generativeai as genai
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image

def get_poppler_path():
    # 1. Check environment variable
    env_path = os.environ.get("POPPLER_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
        
    # 2. Check the hardcoded local Windows path (dev fallback)
    local_path = r"C:\Users\yashs\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin"
    if os.path.exists(local_path):
        return local_path

    # 3. Return None to let pdf2image use system PATH (default for hosted/linux)
    return None

def convert_pdf_to_images(file_input: Union[str, bytes]) -> List[Image.Image]:
    """
    Converts a PDF file (path or bytes) to a list of PIL Images.
    """
    poppler_path = get_poppler_path()
    try:
        if isinstance(file_input, str):
             # It's a file path
            images = convert_from_path(file_input, poppler_path=poppler_path)
        else:
            # It's bytes (from Streamlit upload)
            images = convert_from_bytes(file_input, poppler_path=poppler_path)
        return images
    except Exception as e:
        print(f"Error converting PDF to image: {e}")
        return []

def list_gemini_models(api_key: str) -> List[str]:
    """
    Lists available Gemini models that support content generation.
    """
    if not api_key:
        return []
    try:
        genai.configure(api_key=api_key)
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                # Filter for likely useful models to avoid clutter
                if "gemini" in m.name.lower():
                    models.append(m.name)
        return sorted(models, reverse=True) # Newest first heuristic
    except Exception as e:
        print(f"Error listing models: {e}")
        # Fallback to a couple of reasonable defaults
        return ["models/gemini-2.0-flash", "models/gemini-1.5-flash"]


def _normalize_invoice_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and lightly validate the raw JSON coming back from the model.

    - Ensure required keys exist
    - Normalize currency casing
    - Coerce numeric fields where reasonable
    - Ensure line_items is a list of dicts
    - Ensure summary is a dict with expected keys
    """
    if not isinstance(data, dict):
        return {"error": "Model returned non-object JSON"}

    # Ensure top-level keys exist
    for key in [
        "invoice_number",
        "date",
        "currency",
        "total_amount",
        "vendor_name",
        "vendor_address",
        "vendors_gst_number",
        "line_items",
        "summary",
    ]:
        if key == "line_items":
            data.setdefault(key, [])
        elif key == "summary":
            data.setdefault(key, {})
        else:
            data.setdefault(key, None)

    # Normalize currency (ISO-like upper-case)
    if isinstance(data.get("currency"), str):
        data["currency"] = data["currency"].strip().upper()

    # Coerce total_amount to float when possible
    total_amount = data.get("total_amount")
    if isinstance(total_amount, str):
        try:
            data["total_amount"] = float(total_amount.replace(",", "").strip())
        except ValueError:
            pass

    # Normalize summary section
    summary = data.get("summary") or {}
    if not isinstance(summary, dict):
        summary = {}
    
    # Ensure summary keys exist
    for key in [
        "subtotal",
        "tax",
        "credits",
        "discounts",
        "charges",
        "billing_period",
        "due_date",
        "account_number",
        "billing_address",
        "bill_to_gst_number",  # customer / Bill To GST / tax ID, lives under summary
    ]:
        summary.setdefault(key, None)
    
    # Coerce numeric fields in summary
    for num_key in ["subtotal", "tax", "credits", "discounts", "charges"]:
        val = summary.get(num_key)
        if isinstance(val, str):
            try:
                summary[num_key] = float(val.replace(",", "").replace("$", "").strip())
            except ValueError:
                pass
    
    data["summary"] = summary

    # Ensure line_items is a list of dicts
    line_items = data.get("line_items") or []
    if not isinstance(line_items, list):
        line_items = [line_items]
    normalized_items: List[Dict[str, Any]] = []
    for item in line_items:
        if not isinstance(item, dict):
            continue
        # Standardize keys
        for k in ["description", "quantity", "unit_price", "total"]:
            item.setdefault(k, None)

        # Numeric coercion for quantity/unit_price/total
        for num_key in ["quantity", "unit_price", "total"]:
            val = item.get(num_key)
            if isinstance(val, str):
                try:
                    item[num_key] = float(val.replace(",", "").strip())
                except ValueError:
                    continue
        normalized_items.append(item)

    data["line_items"] = normalized_items

    return data


def _is_reasonable_invoice(data: Dict[str, Any]) -> bool:
    """
    Lightweight validation to detect obviously bad parses so we can retry once.

    Heuristics:
    - Must have at least 1 line_item
    - If both summary.subtotal and per-line totals are numeric, they should be
      within a loose range of each other.
    """
    if not isinstance(data, dict):
        return False

    line_items = data.get("line_items") or []
    if not isinstance(line_items, list) or len(line_items) == 0:
        return False

    summary = data.get("summary") or {}
    if not isinstance(summary, dict):
        return True

    subtotal = summary.get("subtotal")
    if not isinstance(subtotal, (int, float)):
        return True

    total_from_items = 0.0
    has_numeric_total = False
    for item in line_items:
        if not isinstance(item, dict):
            continue
        t = item.get("total")
        if isinstance(t, (int, float)):
            total_from_items += float(t)
            has_numeric_total = True

    if not has_numeric_total:
        return True

    if subtotal <= 0:
        return True

    ratio = total_from_items / subtotal
    # Accept a generous band; outside means likely bad parse
    return 0.5 <= ratio <= 1.5


def validate_invoice(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a set of validation checks on a normalized invoice JSON.

    Returns:
        {
          "status": "valid" | "suspicious" | "failed",
          "issues": [ "text description of problem", ... ]
        }
    """
    issues: List[str] = []

    if not isinstance(data, dict):
        return {
            "status": "failed",
            "issues": ["Invoice data is not an object."],
            "message": "Invoice appears invalid. Please treat it as fake until manually reviewed.",
        }

    summary = data.get("summary") or {}
    line_items = data.get("line_items") or []

    # 1) Required top-level fields present
    for field in ["invoice_number", "date", "currency", "total_amount", "vendor_name"]:
        if not data.get(field):
            issues.append(f"Missing or empty required field: {field}.")

    # 2) Summary + line items presence
    if not isinstance(summary, dict):
        issues.append("Summary section is missing or not an object.")
    if not isinstance(line_items, list) or len(line_items) == 0:
        issues.append("No line items detected.")

    # 3) Basic type / format checks
    total_amount = data.get("total_amount")
    if total_amount is not None and not isinstance(total_amount, (int, float)):
        issues.append("Total amount is not numeric.")

    from datetime import datetime

    def _check_date(name: str, value: Any) -> None:
        if value in (None, "", "null"):
            return
        if not isinstance(value, str):
            issues.append(f"{name} is not a string date.")
            return
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                datetime.strptime(value, fmt)
                return
            except Exception:
                continue
        issues.append(f"{name} is not a valid date: {value!r}.")

    _check_date("date", data.get("date"))
    _check_date("due_date", summary.get("due_date") if isinstance(summary, dict) else None)

    currency = data.get("currency")
    if currency and (not isinstance(currency, str) or len(currency.strip()) != 3):
        issues.append(f"Currency should be a 3-letter code (e.g., USD, EUR), got {currency!r}.")

    # 4) Amount consistency (line items vs subtotal vs total)
    subtotal = summary.get("subtotal") if isinstance(summary, dict) else None
    if isinstance(subtotal, (int, float)) and isinstance(line_items, list) and line_items:
        total_from_items = 0.0
        for item in line_items:
            if not isinstance(item, dict):
                continue
            t = item.get("total")
            if isinstance(t, (int, float)):
                total_from_items += float(t)

        if subtotal > 0 and total_from_items > 0:
            ratio = total_from_items / float(subtotal)
            if not (0.9 <= ratio <= 1.1):
                issues.append(
                    f"Line item totals ({total_from_items}) differ significantly from subtotal ({subtotal})."
                )

    if isinstance(subtotal, (int, float)) and isinstance(total_amount, (int, float)):
        tax = summary.get("tax") if isinstance(summary, dict) else None
        charges = summary.get("charges") if isinstance(summary, dict) else None
        discounts = summary.get("discounts") if isinstance(summary, dict) else None
        credits = summary.get("credits") if isinstance(summary, dict) else None

        def _val(x: Any) -> float:
            return float(x) if isinstance(x, (int, float)) else 0.0

        expected_total = subtotal + _val(tax) + _val(charges) - _val(discounts) - _val(credits)
        if expected_total > 0:
            ratio = float(total_amount) / expected_total
            if not (0.9 <= ratio <= 1.1):
                issues.append(
                    f"Total amount ({total_amount}) is inconsistent with subtotal/tax/discounts ({expected_total})."
                )

    # 5) Simple sanity checks on line items
    if isinstance(line_items, list):
        for idx, item in enumerate(line_items):
            if not isinstance(item, dict):
                issues.append(f"Line item {idx} is not an object.")
                continue
            if not item.get("description"):
                issues.append(f"Line item {idx} has no description.")
            qty = item.get("quantity")
            if isinstance(qty, (int, float)) and qty < 0:
                issues.append(f"Line item {idx} has negative quantity ({qty}).")

    # Decide status based on number/severity of issues
    if not issues:
        status = "valid"
        message = "Invoice verified successfully."
    else:
        # For now treat any issue as suspicious; callers can decide how strict to be
        # If there are clearly critical issues (no line items, no totals), mark failed.
        critical = any(
            "No line items detected" in msg
            or "Total amount is not numeric" in msg
            or "Summary section is missing" in msg
            for msg in issues
        )
        status = "failed" if critical else "suspicious"
        if status == "failed":
            message = "Invoice appears invalid or fake. Please review before trusting this data."
        else:
            message = "Invoice looks suspicious. Please review the flagged issues."

    return {"status": status, "issues": issues, "message": message}

def extract_invoice_data(
    image: Union[Image.Image, List[Image.Image]],
    api_key: str,
    provider: str = "Gemini",
    model_name: str = "gemini-2.0-flash",
) -> Optional[Dict[str, Any]]:
    """
    Sends the image(s) to the selected provider (Gemini, OpenAI, or Claude) to extract invoice data.

    `image` can be a single PIL Image or a list of images (for multi-page PDFs).
    """
    if not api_key:
        return {"error": "API Key is missing."}

    # Clean and validate API key
    api_key = api_key.strip()
    
    # Basic validation for OpenAI/Claude keys
    if provider == "OpenAI" and not api_key.startswith("sk-"):
        return {"error": "Invalid OpenAI API key format. OpenAI keys should start with 'sk-'."}
    if provider == "Claude" and not api_key.startswith("sk-"):
        return {"error": "Invalid Claude API key format. Claude keys should start with 'sk-'."}

    try:
        # Normalize to list internally for easier handling
        images: List[Image.Image]
        if isinstance(image, list):
            images = image
        else:
            images = [image]

        if provider == "OpenAI":
            from openai import OpenAI
            import base64
            
            client = OpenAI(api_key=api_key)

            base_text_instructions = (
                "Extract invoice data into JSON with these fields:\n"
                "- invoice_number (string)\n"
                "- date (YYYY-MM-DD)\n"
                "- currency (ISO code like USD, EUR)\n"
                "- total_amount (number)\n"
                "- vendor_name (string)\n"
                "- vendor_address (string)\n"
                "- vendors_gst_number (string, vendor GSTIN/VAT or other vendor tax ID)\n"
                "- summary: {subtotal, tax, credits, discounts, charges, billing_period, due_date, account_number, billing_address, bill_to_gst_number}\n"
                "- line_items: list of {description, quantity, unit_price, total}\n\n"
                "For line_items, treat each visible row in the main charges/usage table as exactly one line item.\n"
                "Do NOT merge, split, invent, or drop rows. Copy numbers exactly; if any value is unreadable, set it to null instead of guessing.\n"
                "Look carefully around the vendor name for address blocks labeled 'Address', 'Registered Office', or similar.\n"
                "Look carefully for GST/tax labels such as 'GST', 'GSTIN', 'GST No', 'GST Number', 'VAT', or other tax IDs.\n"
                "When there are two GST numbers on the invoice, map the one near the vendor block to 'vendors_gst_number' and the one near the 'Bill to' block to 'bill_to_gst_number'.\n"
                "If a field truly cannot be found anywhere on the invoice, set it to null. "
                "If there is a plausible candidate on the page, return your best guess instead of null.\n"
                "Respond with strictly valid JSON only, no extra text."
            )

            # OpenAI expects base64 images; support multiple pages
            def _call_openai_once() -> Dict[str, Any]:
                content_parts: List[Dict[str, Any]] = [
                    {
                        "type": "text",
                        "text": base_text_instructions,
                    }
                ]

                for img in images:
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_str}",
                            },
                        }
                    )

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=[
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    ],
                    response_format={"type": "json_object"}
                )
                raw = response.choices[0].message.content
                try:
                    data = json.loads(raw)
                except Exception:
                    # Some SDK variants already return a dict
                    data = raw  # type: ignore[assignment]
                normalized = _normalize_invoice_json(data)
                normalized["validation"] = validate_invoice(normalized)
                return normalized

            try:
                # Single retry with simple validation to reduce inconsistent parses
                last_result: Dict[str, Any] = {}
                for attempt in range(2):
                    last_result = _call_openai_once()
                    if _is_reasonable_invoice(last_result) or attempt == 1:
                        return last_result
            except Exception as openai_error:
                # Provide more helpful error messages
                error_str = str(openai_error)
                error_lower = error_str.lower()
                
                if "401" in error_str or "invalid_api_key" in error_lower:
                    return {
                        "error": (
                            "Invalid OpenAI API key. Please check:\n"
                            "1. Your API key is correct and active at https://platform.openai.com/account/api-keys\n"
                            "2. The key has sufficient credits\n"
                            "3. The key is properly set in your .env file without quotes or extra spaces"
                        )
                    }
                elif "429" in error_str:
                    # 429 can mean quota exceeded OR rate limiting
                    if "insufficient_quota" in error_lower or "quota" in error_lower:
                        return {
                            "error": (
                                "OpenAI API quota/billing issue detected.\n\n"
                                "Even if your dashboard shows budget available, OpenAI may require:\n"
                                "1. A payment method on file (even for free credits) - https://platform.openai.com/account/billing\n"
                                "2. Account verification/activation\n"
                                "3. Organization-level limits may apply\n\n"
                                "💡 Quick fix: Switch to 'Gemini' provider in the sidebar (free tier, no billing required)"
                            )
                    }
                    else:
                        # Rate limiting
                        return {
                            "error": (
                                "OpenAI API rate limit exceeded. Too many requests.\n"
                                "Please wait a moment and try again, or switch to 'Gemini' provider."
                            )
                        }
                return {"error": f"OpenAI API error: {error_str}"}
                return {"error": f"OpenAI API error: {error_str}"}

        elif provider == "Claude":
            import anthropic
            import base64
            
            client = anthropic.Anthropic(api_key=api_key)

            def _call_claude_once() -> Dict[str, Any]:
                content_parts: List[Dict[str, Any]] = []

                # 1️⃣ Images first
                for img in images:
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    content_parts.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_str,
                            },
                        }
                    )

                # 2️⃣ Clear + strict instructions
                content_parts.append(
                    {
                        "type": "text",
                        "text": (
                            "You are an expert invoice extraction AI.\\n\\n"
                            "Extract invoice data into a SINGLE valid JSON object with EXACTLY this schema:\\n\\n"
                            "{\\n"
                            '  "invoice_number": string | null,\\n'
                            '  "date": "YYYY-MM-DD" | null,\\n'
                            '  "currency": string | null,\\n'
                            '  "total_amount": number | null,\\n'
                            '  "vendor_name": string | null,\\n'
                            '  "vendor_address": string | null,\\n'
                            '  "vendors_gst_number": string | null,\\n'
                            '  "summary": {\\n'
                            '    "subtotal": number | null,\\n'
                            '    "tax": number | null,\\n'
                            '    "credits": number | null,\\n'
                            '    "discounts": number | null,\\n'
                            '    "charges": number | null,\\n'
                            '    "billing_period": string | null,\\n'
                            '    "due_date": "YYYY-MM-DD" | null,\\n'
                            '    "account_number": string | null,\\n'
                            '    "billing_address": string | null,\\n'
                            '    "bill_to_gst_number": string | null\\n'
                            '  },\\n'
                            '  "line_items": [\\n'
                            '    { "description": string | null, "quantity": number | null, "unit_price": number | null, "total": number | null }\\n'
                            '  ]\\n'
                            "}\\n\\n"
                            "Rules:\\n"
                            "- Do NOT guess values.\\n"
                            "- If unreadable or missing, use null.\\n"
                            "- Copy numbers exactly as shown.\\n"
                            "- One row in the invoice table = one line_item.\\n"
                            "- Respond with ONLY valid JSON. No markdown. No explanation."
                        ),
                    }
                )

                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=4096,
                    temperature=0,
                    messages=[
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    ],
                )

                raw = response.content[0].text

                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0].strip()
                elif "```" in raw:
                    raw = raw.split("```")[1].split("```")[0].strip()

                data = json.loads(raw)
                normalized = _normalize_invoice_json(data)
                normalized["validation"] = validate_invoice(normalized)
                return normalized

            try:

                last_result: Dict[str, Any] = {}
                for attempt in range(2):
                    last_result = _call_claude_once()
                    if _is_reasonable_invoice(last_result) or attempt == 1:
                        return last_result
            except Exception as e:
                return {"error": f"Claude API error: {str(e)}"}
        else:
            # Default to Gemini
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel(
                    model_name,
                    generation_config={
                        # Ask Gemini to emit JSON only
                        "response_mime_type": "application/json",
                        "temperature": 0,
                    },
                )
            except Exception:
                # Fallback if specific model fails (though listing showed it exists)
                model = genai.GenerativeModel(
                    "gemini-2.0-flash",
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0,
                    },
                )

            prompt = """
You are an expert invoice extraction AI.

Goal:
Return a single JSON object with the following exact schema:
{
  "invoice_number": "string or null",
  "date": "YYYY-MM-DD or null",
  "currency": "string (ISO 4217 code like USD, EUR) or null",
  "total_amount": number or null,
  "vendor_name": "string or null",
  "vendor_address": "string or null",
  "vendors_gst_number": "string or null",
  "summary": {
    "subtotal": number or null,
    "tax": number or null,
    "credits": number or null,
    "discounts": number or null,
    "charges": number or null,
    "billing_period": "string or null",
    "due_date": "YYYY-MM-DD or null",
    "account_number": "string or null",
    "billing_address": "string or null",
    "bill_to_gst_number": "string or null"
  },
  "line_items": [
    {
      "description": "string or null",
      "quantity": number or null,
      "unit_price": number or null,
      "total": number or null
    }
  ]
}

Instructions:
- If you cannot confidently find a field, set it to null instead of guessing.
- Use the grand total including tax for total_amount if available.
- If multiple dates exist, use the invoice issue date.
- Extract all summary information like subtotal, tax, credits, discounts from the invoice.
- Carefully extract the vendor's full address and any GST/tax identification numbers (e.g., GSTIN, VAT) if present.\n
- Look near the vendor name for address blocks labeled 'Address', 'Registered Office', or similar.\n
- Look for GST/tax labels such as 'GST', 'GSTIN', 'GST No', 'GST Number', 'VAT', or other tax IDs.\n
- When there are two GST numbers on the invoice, map the one associated with the vendor (e.g., near the company name like 'Exafunction, Inc.' or 'From' section) to 'vendors_gst_number', and the one in the 'Bill to' or customer section to summary.bill_to_gst_number (place it inside the summary object, under the billing_address).\n
- If any address or GST/tax ID appears anywhere on the invoice, do NOT return null for vendor_address, vendors_gst_number, or summary.bill_to_gst_number; instead, return the best-matching value you can find.
- For billing_period, extract the date range if shown (e.g., "July 1 - July 31, 2014").
- Respond with strictly valid JSON only, with no markdown and no extra text.
""".strip()

            def _call_gemini_once() -> Dict[str, Any]:
                # Feed all pages/images to Gemini in a single call
                contents: List[Any] = [prompt]
                contents.extend(images)

                response = model.generate_content(contents)

                # With response_mime_type="application/json", .text should be pure JSON
                text = response.text.strip()
                data = json.loads(text)

                normalized = _normalize_invoice_json(data)
                normalized["validation"] = validate_invoice(normalized)
                return normalized

            # Single retry with simple validation to reduce inconsistent parses
            last_result = {}
            for attempt in range(2):
                last_result = _call_gemini_once()
                if _is_reasonable_invoice(last_result) or attempt == 1:
                    return last_result

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Test block
    print("Run via Streamlit app or import functions.")
