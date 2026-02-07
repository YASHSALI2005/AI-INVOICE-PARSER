from typing import Dict, Any, Union, List, Optional
from PIL import Image, ImageFilter
import numpy as np
import google.generativeai as genai
import extractor
import json

def check_blur(image: Image.Image, threshold: float = 100.0) -> Dict[str, Any]:
    """
    Detects if an image is blurry using the Variance of Laplacian method.
    Returns: {"is_blurry": bool, "score": float}
    """
    try:
        # Convert to grayscale
        gray = image.convert("L")
        # Convert to numpy array
        img_array = np.array(gray)
        
        # Laplacian Kernel
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        
        # We can implement simple convolution or use PIL's Filter
        # Using PIL's built-in filter is safer if we want to avoid custom convolution code
        # but PIL's FIND_EDGES is not exactly Laplacian. 
        # Let's use Pillow's Kernel filter for exact Laplacian.
        
        laplacian_img = gray.filter(ImageFilter.Kernel((3, 3), kernel.flatten(), scale=1, offset=0))
        lap_array = np.array(laplacian_img)
        
        # Calculate variance
        score = lap_array.var()
        
        # Threshold: usually < 100 is considered blurry for text documents
        is_blurry = score < threshold
        
        return {"is_blurry": is_blurry, "score": score}
        
    except Exception as e:
        print(f"Error checking blur: {e}")
        return {"is_blurry": False, "score": 9999.0} # Fail safe


def verify_and_correct_json(
    image: Union[Image.Image, List[Image.Image]],
    initial_json: Dict[str, Any],
    api_key: str,
    provider="Claude",  
    model_name="claude-3-haiku-20240307"
) -> Dict[str, Any]:
    """
    Step 2: Verification.
    Feeds the Image + Initial JSON back to the LLM to verify and correct discrepancies.
    """
    if not api_key:
        return {"error": "API Key is missing."}

    if provider == "Claude":
        import anthropic
        import io
        import base64
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Convert initial JSON to string for the prompt
        json_str = json.dumps(initial_json, indent=2)
        
        prompt = f"""
        You are an expert Invoice Verification AI.
    
        Your task is to VERIFY and CORRECT the extraction of invoice data.
        
        1.  **Input**:
            *   An image of an invoice.
            *   A JSON object representing the data already extracted from this invoice (see below).
    
        2.  **Action**:
            *   Carefully compare every field in the provided JSON against the actual image.
            *   If a value in the JSON is INCORRECT or MISSING (null) but matches the image, FIX it.
            *   If a value is correct, keep it.
            *   Pay special attention to:
                *   Invoice Number
                *   Dates (Format YYYY-MM-DD)
                *   Total Amounts and subtotals.
                *   Vendor details (Name, Address, GST/Tax IDs).
                *   Line items (ensure quantities and totals match).
    
        3.  **Strict Output**:
            *   Return the FINAL, CORRECTED JSON object.
            *   Do NOT return a diff or explanation. ONLY the valid JSON.
    
        ----- EXISTING EXTRACTED DATA -----
        {json_str}
        -----------------------------------
        """

        content_parts = []
        if isinstance(image, list):
            imgs = image
        else:
            imgs = [image]
            
        for img in imgs:
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

        try:
            response = client.messages.create(
                model=model_name or "claude-3-haiku-20240307",
                max_tokens=4096,
                temperature=0,
                messages=[{"role": "user", "content": content_parts}]
            )
            raw = response.content[0].text
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            
            corrected_json = json.loads(raw)
            return extractor._normalize_invoice_json(corrected_json)
        except Exception as e:
            return {"error": f"Verification Step Failed: {str(e)}", "partial_data": initial_json}

    # Default to Gemini
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel(
            model_name,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0,
            },
        )
    except Exception:
         model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0,
            },
        )

    # Convert initial JSON to string for the prompt
    json_str = json.dumps(initial_json, indent=2)

    prompt = f"""
    You are an expert Invoice Verification AI.

    Your task is to VERIFY and CORRECT the extraction of invoice data.
    
    1.  **Input**:
        *   An image of an invoice.
        *   A JSON object representing the data already extracted from this invoice (see below).

    2.  **Action**:
        *   Carefully compare every field in the provided JSON against the actual image.
        *   If a value in the JSON is INCORRECT or MISSING (null) but matches the image, FIX it.
        *   If a value is correct, keep it.
        *   Pay special attention to:
            *   Invoice Number
            *   Dates (Format YYYY-MM-DD)
            *   Total Amounts and subtotals.
            *   Vendor details (Name, Address, GST/Tax IDs).
            *   Line items (ensure quantities and totals match).

    3.  **Strict Output**:
        *   Return the FINAL, CORRECTED JSON object.
        *   Do NOT return a diff or explanation. ONLY the valid JSON.

    ----- EXISTING EXTRACTED DATA -----
    {json_str}
    -----------------------------------
    """

    try:
        # Prepare contents
        contents: List[Any] = [prompt]
        if isinstance(image, list):
            contents.extend(image)
        else:
            contents.append(image)

        response = model.generate_content(contents)
        text = response.text.strip()
        corrected_json = json.loads(text)
        
        # Normalize just in case
        return extractor._normalize_invoice_json(corrected_json)

    except Exception as e:
        return {"error": f"Verification Step Failed: {str(e)}", "partial_data": initial_json}

def extract_with_verification(
    file_input: Union[str, bytes],
    api_key: str,
    provider: str = "Claude",
    model_name: str = "claude-3-haiku-20240307",
) -> Dict[str, Any]:
    """
    Orchestrates the 2-step process:
    1. PDF -> Image -> Extract (Step 1)
    2. Image + JSON -> Verify & Correct (Step 2)
    """
    
    # 1. Convert/Load Images
    images = []
    if isinstance(file_input, str):
        ext = file_input.lower()
        if ext.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            try:
                img = Image.open(file_input)
                # Force load to avoid resource issues
                img.load()
                images = [img]
            except Exception as e:
                return {"error": f"Failed to load image: {str(e)}"}
        else:
             # Assume PDF
             images = extractor.convert_pdf_to_images(file_input)
    else:
        # Bytes input (assume PDF bytes for now, or could inspect header)
        images = extractor.convert_pdf_to_images(file_input)

    if not images:
        return {"error": "Failed to convert input to images."}

    # 1.5 Check for Blur
    # We check the first page as a proxy, or all pages. Let's check the first page.
    blur_result = check_blur(images[0])
    if blur_result["is_blurry"]:
        return {
            "error": "Blurry Image Detected",
            "message": f"The uploaded image appears to be blurry (Score: {blur_result['score']:.2f}). Please re-upload a clear PDF or Image to ensure accurate extraction."
        }
    print(f"Blur Check Passed: Score {blur_result['score']:.2f}")

    # 2. Step 1: Initial Extraction
    print("--- Step 1: Initial Extraction ---")
    initial_data = extractor.extract_invoice_data(
        images, 
        api_key=api_key, 
        provider=provider, 
        model_name=model_name
    )

    if "error" in initial_data:
        return initial_data

    # 3. Step 2: Verification Loop
    print("--- Step 2: Verification & Correction ---")
    final_data = verify_and_correct_json(
        images,
        initial_data,
        api_key,
        provider=provider,
        model_name=model_name
    )

    result = {
        "status": "success",
        "initial_extraction": initial_data,
        "final_verified_extraction": final_data
    }
    
    return result
