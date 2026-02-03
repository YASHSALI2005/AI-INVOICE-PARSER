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

    # Ensure keys exist
    for key in [
        "invoice_number",
        "date",
        "currency",
        "total_amount",
        "vendor_name",
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
    for key in ["subtotal", "tax", "credits", "discounts", "charges", "billing_period", "due_date", "account_number", "billing_address"]:
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

def extract_invoice_data(
    image: Union[Image.Image, List[Image.Image]],
    api_key: str,
    provider: str = "Gemini",
    model_name: str = "gemini-2.0-flash",
) -> Optional[Dict[str, Any]]:
    """
    Sends the image(s) to the selected provider (Gemini or OpenAI) to extract invoice data.

    `image` can be a single PIL Image or a list of images (for multi-page PDFs).
    """
    if not api_key:
        return {"error": "API Key is missing."}

    # Clean and validate API key
    api_key = api_key.strip()
    
    # Basic validation for OpenAI keys
    if provider == "OpenAI" and not api_key.startswith("sk-"):
        return {"error": "Invalid OpenAI API key format. OpenAI keys should start with 'sk-'."}

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

            # OpenAI expects base64 images; support multiple pages
            content_parts: List[Dict[str, Any]] = [
                {
                    "type": "text",
                    "text": (
                        "Extract invoice data into JSON with these fields:\n"
                        "- invoice_number (string)\n"
                        "- date (YYYY-MM-DD)\n"
                        "- currency (ISO code like USD, EUR)\n"
                        "- total_amount (number)\n"
                        "- vendor_name (string)\n"
                        "- summary: {subtotal, tax, credits, discounts, charges, billing_period, due_date, account_number, billing_address}\n"
                        "- line_items: list of {description, quantity, unit_price, total}\n\n"
                        "If a field cannot be found, set it to null. "
                        "Respond with strictly valid JSON only, no extra text."
                    ),
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

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
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

                return _normalize_invoice_json(data)
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
            
        else:
            # Default to Gemini
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel(
                    model_name,
                    generation_config={
                        # Ask Gemini to emit JSON only
                        "response_mime_type": "application/json",
                    },
                )
            except Exception:
                # Fallback if specific model fails (though listing showed it exists)
                model = genai.GenerativeModel(
                    "gemini-2.0-flash",
                    generation_config={
                        "response_mime_type": "application/json",
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
  "summary": {
    "subtotal": number or null,
    "tax": number or null,
    "credits": number or null,
    "discounts": number or null,
    "charges": number or null,
    "billing_period": "string or null",
    "due_date": "YYYY-MM-DD or null",
    "account_number": "string or null",
    "billing_address": "string or null"
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
- For billing_period, extract the date range if shown (e.g., "July 1 - July 31, 2014").
- Respond with strictly valid JSON only, with no markdown and no extra text.
""".strip()

            # Feed all pages/images to Gemini in a single call
            contents: List[Any] = [prompt]
            contents.extend(images)

            response = model.generate_content(contents)

            # With response_mime_type="application/json", .text should be pure JSON
            text = response.text.strip()
            data = json.loads(text)

            return _normalize_invoice_json(data)

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Test block
    print("Run via Streamlit app or import functions.")
