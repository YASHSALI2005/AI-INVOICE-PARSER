import os
import json
from typing import Dict, Any

from dotenv import load_dotenv

import extractor


def compare_fields(expected: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, bool]:
    """
    Compare a few key fields between expected invoice2data JSON
    and the LLM extractor result.
    """
    metrics: Dict[str, bool] = {}

    # Map fields between schemas
    field_mapping = {
        "issuer": "vendor_name",
        "amount": "total_amount",
        "date": "date",
        "invoice_number": "invoice_number",
        "currency": "currency",
    }

    for src_field, llm_field in field_mapping.items():
        exp_val = expected.get(src_field)
        res_val = result.get(llm_field)

        if src_field == "amount":
            try:
                if exp_val is None or res_val is None:
                    metrics["amount"] = False
                else:
                    exp_num = float(exp_val)
                    res_num = float(res_val)
                    metrics["amount"] = abs(exp_num - res_num) < 0.01
            except Exception:
                metrics["amount"] = False
        else:
            metrics[src_field] = (str(exp_val).strip() == str(res_val).strip())

    return metrics


def main() -> None:
    load_dotenv()

    test_dir = os.path.join(
        "invoice2data_repo", "tests", "compare"
    )

    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not gemini_key and not openai_key:
        print("Error: GEMINI_API_KEY or OPENAI_API_KEY must be set.")
        return

    # Prefer Gemini for structured JSON, else fall back to OpenAI
    if gemini_key:
        provider = "Gemini"
        api_key = gemini_key
    else:
        provider = "OpenAI"
        api_key = openai_key  # type: ignore[assignment]

    print(f"Using provider: {provider}")
    print(f"Scanning test directory: {os.path.abspath(test_dir)}")

    pdf_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found.")
        return

    overall_field_correct = {
        "issuer": 0,
        "amount": 0,
        "date": 0,
        "invoice_number": 0,
        "currency": 0,
    }
    overall_field_total = {
        "issuer": 0,
        "amount": 0,
        "date": 0,
        "invoice_number": 0,
        "currency": 0,
    }

    all_fields_correct_count = 0

    for pdf in pdf_files:
        pdf_path = os.path.join(test_dir, pdf)
        json_path = pdf_path.replace(".pdf", ".json")

        print(f"\nProcessing {pdf}...")

        if not os.path.exists(json_path):
            print("  [SKIP] No expected JSON file for comparison.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            expected = json.load(f)

        # Convert PDF to images and run LLM extractor
        with open(pdf_path, "rb") as f_pdf:
            images = extractor.convert_pdf_to_images(f_pdf.read())

        if not images:
            print("  [ERROR] Failed to convert PDF to images.")
            continue

        result = extractor.extract_invoice_data(images, api_key, provider=provider)
        if not result or "error" in result:
            print(f"  [ERROR] Extraction failed: {result.get('error') if isinstance(result, dict) else result}")
            continue

        metrics = compare_fields(expected, result)

        per_invoice_ok = True
        for field, ok in metrics.items():
            # Track totals
            if field == "amount":
                key = "amount"
            else:
                key = field

            overall_field_total[key] += 1
            if ok:
                overall_field_correct[key] += 1
            else:
                per_invoice_ok = False

        if per_invoice_ok:
            all_fields_correct_count += 1

        # Print summary per invoice
        for field, ok in metrics.items():
            status = "OK" if ok else "MISMATCH"
            print(f"  {field}: {status}")

    print("\n=== Summary ===")
    for field, total in overall_field_total.items():
        if total == 0:
            continue
        correct = overall_field_correct[field]
        acc = 100.0 * correct / total
        print(f"{field}: {correct}/{total} correct ({acc:.1f}%)")

    total_invoices = sum(1 for _ in pdf_files)
    print(f"\nInvoices with all tracked fields correct: {all_fields_correct_count}/{total_invoices}")


if __name__ == "__main__":
    main()

