"""
Hybrid Invoice Extractor: Template-based + LLM validation and additional fields.

This combines the best of both worlds:
1. Fast, accurate regex templates for known vendors (invoice2data)
2. LLM validation and extraction of additional fields not in templates
"""

import os
import json
from datetime import datetime, date
from typing import Dict, Any, Optional, List, Union
from PIL import Image

try:
    from invoice2data import extract_data
    from invoice2data.extract.loader import read_templates
    from invoice2data.input import pdftotext
    INVOICE2DATA_AVAILABLE = True
except ImportError:
    INVOICE2DATA_AVAILABLE = False

import extractor


def _json_serializable(val: Any) -> Any:
    """Convert non-JSON-serializable values (datetime, date, tuple) to strings."""
    if val is None:
        return None
    # Any date/datetime-like (including dateutil)
    if hasattr(val, "isoformat") and callable(getattr(val, "isoformat", None)):
        return val.isoformat()
    if isinstance(val, (datetime, date)):
        return val.isoformat() if hasattr(val, "isoformat") else str(val)
    if isinstance(val, tuple):
        # invoice2data sometimes returns date as (year, month, day)
        if len(val) == 3 and all(isinstance(x, (int, float)) for x in val):
            return f"{int(val[0])}-{int(val[1]):02d}-{int(val[2]):02d}"
        return str(val)
    if isinstance(val, (dict, list)):
        return {k: _json_serializable(v) for k, v in val.items()} if isinstance(val, dict) else [_json_serializable(v) for v in val]
    return val


def get_poppler_path():
    """Get Poppler path for invoice2data."""
    return r"C:\Users\yashs\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin"


def extract_with_templates(
    invoice_path: str, template_dir: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Extract invoice data using invoice2data templates (regex-based).
    
    Returns None if no template matches.
    """
    if not INVOICE2DATA_AVAILABLE:
        return None

    try:
        # Check file type - invoice2data templates work best with PDFs
        # For images, we'll skip template extraction and use LLM directly
        file_ext = os.path.splitext(invoice_path)[1].lower()
        if file_ext not in [".pdf"]:
            # Skip template extraction for non-PDF files
            # invoice2data templates are designed for PDF text extraction
            return None

        # Add Poppler to PATH if needed
        poppler_path = get_poppler_path()
        if os.path.exists(poppler_path):
            os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + poppler_path

        # Load templates
        if template_dir is None:
            template_dir = os.path.join(
                "invoice2data_repo", "src", "invoice2data", "extract", "templates"
            )

        if os.path.exists(template_dir):
            templates = read_templates(template_dir)
            # Use pdftotext for PDF files
            result = extract_data(invoice_path, templates=templates, input_module=pdftotext)
            return result if result else None
    except Exception as e:
        # Silently fail - template extraction is optional
        # Error will be logged but not raised
        return None

    return None


def map_template_to_llm_schema(template_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map invoice2data template output to our LLM schema.
    
    Template fields -> LLM fields:
    - issuer/partner_name -> vendor_name
    - amount -> total_amount
    - date -> date (may need format conversion)
    - invoice_number -> invoice_number
    - currency -> currency
    - lines -> line_items (with mapping)
    """
    mapped = {
        "invoice_number": template_data.get("invoice_number"),
        "date": template_data.get("date"),
        "currency": template_data.get("currency"),
        "total_amount": template_data.get("amount"),
        "vendor_name": template_data.get("issuer") or template_data.get("partner_name"),
        "line_items": [],
    }

    # Map line items if present
    if "lines" in template_data and isinstance(template_data["lines"], list):
        for line in template_data["lines"]:
            if isinstance(line, dict):
                mapped["line_items"].append({
                    "description": line.get("name") or line.get("description", ""),
                    "quantity": line.get("qty") or line.get("quantity"),
                    "unit_price": line.get("price_unit") or line.get("unit_price"),
                    "total": line.get("price_subtotal") or line.get("total"),
                })

    # Handle date format conversion
    if mapped["date"]:
        if isinstance(mapped["date"], (datetime, date)):
            mapped["date"] = mapped["date"].strftime("%Y-%m-%d")
        elif isinstance(mapped["date"], tuple):
            # Convert (YYYY, MM, DD) to YYYY-MM-DD
            mapped["date"] = f"{mapped['date'][0]}-{mapped['date'][1]:02d}-{mapped['date'][2]:02d}"
        elif isinstance(mapped["date"], str) and len(mapped["date"]) == 10:
            # Already in YYYY-MM-DD format
            pass
        # Otherwise keep as-is, LLM will handle it

    return mapped


def _call_llm_for_validation(
    image: Union[Image.Image, List[Image.Image]],
    prompt: str,
    api_key: str,
    provider: str = "Gemini",
    model_name: str = "gemini-2.0-flash",
) -> Dict[str, Any]:
    """Call LLM with custom prompt for validation."""
    import google.generativeai as genai
    import io
    import base64
    import google.generativeai as genai
    import io
    import base64
    from openai import OpenAI
    import anthropic

    # Normalize to list
    images: List[Image.Image]
    if isinstance(image, list):
        images = image
    else:
        images = [image]

    try:
        if provider == "OpenAI":
            client = OpenAI(api_key=api_key.strip())
            content_parts = [{"type": "text", "text": prompt}]
            
            for img in images:
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                })

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": content_parts}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)

        elif provider == "Claude":
            client = anthropic.Anthropic(api_key=api_key.strip())
            content_parts = []
            
            for img in images:
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                content_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_str
                    }
                })
            
            content_parts.append({"type": "text", "text": prompt})
            
            response = client.messages.create(
                model=model_name or "claude-3-haiku-20240307",
                max_tokens=4096,
                messages=[{"role": "user", "content": content_parts}]
            )
            
            raw = response.content[0].text
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
                
            return json.loads(raw)

        else:  # Gemini
            genai.configure(api_key=api_key.strip())
            model = genai.GenerativeModel(
                model_name,
                generation_config={"response_mime_type": "application/json"}
            )
            contents = [prompt] + images
            response = model.generate_content(contents)
            return json.loads(response.text.strip())

    except Exception as e:
        return {"error": str(e)}


def extract_additional_fields_with_llm(
    image: Union[Image.Image, List[Image.Image]],
    template_data: Dict[str, Any],
    api_key: str,
    provider: str = "Gemini",
    model_name: str = "gemini-2.0-flash",
) -> Dict[str, Any]:
    """
    Use LLM to:
    1. Validate template-extracted fields
    2. Extract additional fields not in template
    3. Return validation status and additional_info
    """
    # Prepare template summary for LLM (ensure JSON-serializable: no datetime/tuple)
    template_summary = {
        "invoice_number": _json_serializable(template_data.get("invoice_number")),
        "date": _json_serializable(template_data.get("date")),
        "currency": _json_serializable(template_data.get("currency")),
        "total_amount": _json_serializable(template_data.get("total_amount")),
        "vendor_name": _json_serializable(template_data.get("vendor_name")),
        "line_items_count": len(template_data.get("line_items", [])),
    }

    prompt = f"""You are an expert invoice extraction AI. A template-based system has already extracted some fields from this invoice.

Template-extracted fields:
{json.dumps(template_summary, indent=2)}

Your tasks:
1. VALIDATE: Check if the template-extracted fields are correct by examining the invoice image.
2. EXTRACT ADDITIONAL: Find any important fields NOT already extracted by the template.

Return a JSON object with this structure:
{{
  "validation": {{
    "invoice_number": "correct" | "incorrect" | "missing",
    "date": "correct" | "incorrect" | "missing",
    "currency": "correct" | "incorrect" | "missing",
    "total_amount": "correct" | "incorrect" | "missing",
    "vendor_name": "correct" | "incorrect" | "missing"
  }},
  "corrected_fields": {{
    "invoice_number": "actual value if template was wrong, else null",
    "date": "actual value if template was wrong, else null",
    "currency": "actual value if template was wrong, else null",
    "total_amount": "actual value if template was wrong, else null",
    "vendor_name": "actual value if template was wrong, else null"
  }},
  "additional_info": {{
    "tax_amount": "number or null",
    "subtotal": "number or null",
    "due_date": "YYYY-MM-DD or null",
    "payment_terms": "string or null",
    "billing_address": "string or null",
    "shipping_address": "string or null",
    "purchase_order": "string or null",
    "notes": "string or null",
    "other_fields": {{}}
  }}
}}

If template fields are correct, set corrected_fields to null for those fields.
Extract any additional useful information in additional_info.
"""

    try:
        llm_result = _call_llm_for_validation(
            image, prompt, api_key, provider, model_name
        )

        if "error" in llm_result:
            return {
                "validation": {k: "unknown" for k in template_summary.keys()},
                "corrected_fields": {},
                "additional_info": {},
            }

        return llm_result

    except Exception as e:
        return {
            "validation": {k: "error" for k in template_summary.keys()},
            "corrected_fields": {},
            "additional_info": {},
            "error": str(e),
        }


def hybrid_extract_invoice(
    invoice_path: str,
    image: Union[Image.Image, List[Image.Image]],
    api_key: str,
    provider: str = "Gemini",
    model_name: str = "gemini-2.0-flash",
    template_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Hybrid extraction: Try templates first, then use LLM for validation and additional fields.
    
    Returns combined result with:
    - Template-extracted fields (if template matched)
    - LLM validation results
    - Additional fields from LLM
    """
    result: Dict[str, Any] = {
        "extraction_method": "hybrid",
        "template_matched": False,
        "template_data": {},
        "llm_validation": {},
        "additional_info": {},
        "final_data": {},
    }

    # Step 1: Try template extraction
    template_data = extract_with_templates(invoice_path, template_dir)
    
    if template_data:
        result["template_matched"] = True
        result["template_data"] = template_data
        mapped_template = map_template_to_llm_schema(template_data)
        result["final_data"] = mapped_template.copy()
    else:
        result["template_matched"] = False
        # No template match, use LLM for full extraction
        llm_result = extractor.extract_invoice_data(
            image, api_key, provider=provider, model_name=model_name
        )
        if "error" not in llm_result:
            result["final_data"] = llm_result
        else:
            result["final_data"] = {"error": llm_result.get("error")}
        return result

    # Step 2: Use LLM to validate and extract additional fields
    llm_analysis = extract_additional_fields_with_llm(
        image, mapped_template, api_key, provider, model_name
    )

    result["llm_validation"] = llm_analysis.get("validation", {})
    result["additional_info"] = llm_analysis.get("additional_info", {})

    # Step 3: Apply corrections if LLM found errors
    corrected = llm_analysis.get("corrected_fields", {})
    for field, value in corrected.items():
        if value is not None:
            result["final_data"][field] = value

    # Step 4: Add additional_info to final_data
    if result["additional_info"]:
        result["final_data"]["additional_info"] = result["additional_info"]

    return result
