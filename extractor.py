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
    Raises RuntimeError with a helpful message if conversion fails (e.g. poppler not installed).
    """
    poppler_path = get_poppler_path()
    try:
        if isinstance(file_input, str):
            images = convert_from_path(file_input, poppler_path=poppler_path)
        else:
            images = convert_from_bytes(file_input, poppler_path=poppler_path)
        return images if images else []
    except Exception as e:
        err = str(e).strip() or repr(e)
        if "pdftoppm" in err.lower() or "poppler" in err.lower() or "Unable to get page count" in err:
            raise RuntimeError(
                "PDF conversion failed: Poppler is required but not found. "
                "On macOS run: brew install poppler. On Ubuntu/Debian: sudo apt install poppler-utils."
            ) from e
        raise RuntimeError(f"PDF conversion failed: {err}") from e

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
                if "gemini" in m.name.lower():
                    models.append(m.name)
        return sorted(models, reverse=True) # Newest first heuristic
    except Exception as e:
        print(f"Error listing models: {e}")
        return ["models/gemini-2.0-flash", "models/gemini-1.5-flash"]


def _get_prompt_for_type(document_type: str, provider: str) -> str:
    if document_type == "Purchase Order bill":
        schema = '''{
  "from": "string or null",
  "to": "string or null",
  "po_number": "string or null",
  "date_of_issue": "YYYY-MM-DD or null",
  "delivery_expected_by": "YYYY-MM-DD or null",
  "summary": { "total_commitment": 0.0 },
  "line_items": [ { "item_description": "string or null", "quantity": 0.0, "unit_price": 0.0, "tax_amount": 0.0, "total": 0.0 } ]
}'''
        rules = """- One row in the main table = one line_item.
- IMPORTANT: Extract 'tax_amount' ONLY if it is literally printed on that specific line. Do NOT calculate it yourself."""
    elif document_type == "Sales Order bill":
        schema = '''{
  "seller": "string or null",
  "buyer": "string or null",
  "order_number": "string or null",
  "order_date": "YYYY-MM-DD or null",
  "status": "string or null (Confirmed or Pending Fulfillment)",
  "summary": { "order_subtotal": 0.0, "estimated_shipping": 0.0,"tax_amount": 0.0, "order_total": 0.0 },
  "order_items": [ { "product_code": "string or null", "description": "string or null", "qty": 0.0, "unit_price": 0.0 } ]
}'''
        rules = """- One row in the main table = one order_item.
- IMPORTANT: Extract 'tax_amount' from the summary section matching the document total tax. Do NOT calculate it yourself."""
    else: # Sales Invoice bill
        schema = '''{
  "from": "string or null",
  "bill_to": "string or null",
  "invoice_number": "string or null",
  "date_of_issue": "YYYY-MM-DD or null",
  "payment_due_date": "YYYY-MM-DD or null",
  "summary": { "total_usage_sale_charges": 0.0, "tax_percentage": 0.0, "total_amount_due": 0.0 },
  "charges": [ { "description": "string or null", "quantity_hours": 0.0, "rate": 0.0, "amount": 0.0 } ]
}'''
        rules = "- One row in the main billable services table = one charge item."

    if provider in ["Claude", "OpenAI"]:
        return f"Extract {document_type} data into a SINGLE JSON object with EXACTLY this schema:\n{schema}\n\nRules:\n- Do NOT guess values. If unreadable or missing, use null.\n- Copy numbers exactly as shown.\n{rules}\n- Respond with ONLY valid JSON. No markdown. No explanation."
    else: # Gemini
        return f"You are an expert extraction AI.\n\nGoal:\nReturn a single JSON object for a {document_type} with the following exact schema:\n{schema}\n\nInstructions:\n- If you cannot confidently find a field, set it to null instead of guessing.\n- If multiple dates exist, use the issue date.\n- Extract all summary information from the document.\n{rules}\n- Respond with strictly valid JSON only, with no markdown and no extra text."


def _normalize_invoice_json(data: Dict[str, Any], document_type: str = "Sales Invoice bill") -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"error": "Model returned non-object JSON"}

    def _coerce_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in list(d.items()):
            if isinstance(v, str):
                try:
                    cval = v.replace(",", "").replace(" ", "").replace("$", "").replace("€", "").replace("£", "")
                    if cval and (cval[0].isdigit() or (cval.startswith("-") and cval[1:].isdigit())):
                        # If it just contains digits and maybe a dot
                        if cval.replace(".", "", 1).isdigit() or cval.replace("-", "", 1).replace(".", "", 1).isdigit():
                            d[k] = float(cval)
                        else:
                            # If it's something like "0.00 (calculated)" or has noise
                            # Keep it as string if it doesn't look like a simple number
                            pass
                except ValueError:
                    pass
            elif isinstance(v, dict):
                d[k] = _coerce_dict(v)
            elif isinstance(v, list):
                new_list = []
                for item in v:
                    if isinstance(item, dict):
                        new_list.append(_coerce_dict(item))
                    else:
                        new_list.append(item)
                d[k] = new_list
        return d
        
    data = _coerce_dict(data)

    if "summary" not in data or not isinstance(data["summary"], dict):
        data["summary"] = {}

    list_keys = ["line_items", "charges", "order_items"]
    for lk in list_keys:
        if lk in data and not isinstance(data[lk], list):
            data[lk] = [data[lk]]

    return data


def _is_reasonable_invoice(data: Dict[str, Any], document_type: str = "Sales Invoice bill") -> bool:
    if not isinstance(data, dict):
        return False
    # Check if there's any list with items
    has_list = any(isinstance(data.get(k), list) and len(data.get(k)) > 0 for k in ["line_items", "charges", "order_items"])
    if not has_list:
        return False
    return True
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


def extract_invoice_data(
    image: Union[Image.Image, List[Image.Image]],
    api_key: str,
    provider: str = "Gemini",
    model_name: str = "gemini-2.0-flash",
    document_type: str = "Sales Invoice bill",
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

            base_text_instructions = _get_prompt_for_type(document_type, "OpenAI")

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
                normalized = _normalize_invoice_json(data, document_type)
                return normalized

            try:
                # Single retry with simple validation to reduce inconsistent parses
                last_result: Dict[str, Any] = {}
                for attempt in range(2):
                    last_result = _call_openai_once()
                    if _is_reasonable_invoice(last_result, document_type) or attempt == 1:
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
                        "text": _get_prompt_for_type(document_type, "Claude")
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
                normalized = _normalize_invoice_json(data, document_type)
                return normalized

            try:

                last_result: Dict[str, Any] = {}
                for attempt in range(2):
                    last_result = _call_claude_once()
                    if _is_reasonable_invoice(last_result, document_type) or attempt == 1:
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

            prompt = _get_prompt_for_type(document_type, "Gemini")

            def _call_gemini_once() -> Dict[str, Any]:
                # Feed all pages/images to Gemini in a single call
                contents: List[Any] = [prompt]
                contents.extend(images)

                response = model.generate_content(contents)

                # With response_mime_type="application/json", .text should be pure JSON
                text = response.text.strip()
                data = json.loads(text)

                normalized = _normalize_invoice_json(data, document_type)
                return normalized

            # Single retry with simple validation to reduce inconsistent parses
            last_result = {}
            for attempt in range(2):
                last_result = _call_gemini_once()
                if _is_reasonable_invoice(last_result, document_type) or attempt == 1:
                    return last_result

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Test block
    print("Run via Streamlit app or import functions.")
