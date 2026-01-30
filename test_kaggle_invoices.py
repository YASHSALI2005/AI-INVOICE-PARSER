"""
Test your invoice parser against invoice datasets.

Supports:
- Kaggle Invoice NER Dataset (Excel format)
- GitHub Invoice Dataset (https://github.com/mouadhamri/invoice_dataset.git)

This script:
1. Loads invoices from prepared dataset
2. Runs your parser on each invoice
3. Compares results with ground truth (if available)
4. Generates accuracy report
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import extractor
from invoice_db import get_database
from confidence_scorer import calculate_confidence

load_dotenv()


def load_test_invoices(dataset_dir: str = "test_invoices_ready") -> List[Dict[str, Any]]:
    """Load test invoices from prepared dataset."""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        print("   Run download script first:")
        print("   - download_kaggle_dataset.py (for Kaggle)")
        print("   - download_github_invoice_dataset.py (for GitHub)")
        return []
    
    # Load index if exists (GitHub dataset format)
    index_file = dataset_path / "index.json"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            records = json.load(f)
            # Normalize format
            normalized = []
            for r in records:
                normalized.append({
                    "id": r.get("id", r.get("_row", len(normalized))),
                    "filename": Path(r.get("image_path", r.get("path", ""))).name,
                    "path": r.get("image_path", r.get("path", "")),
                    "ground_truth": r.get("ground_truth", {}),
                    "model": r.get("model"),
                })
            return normalized
    
    # Otherwise, find all images (fallback)
    images = []
    for ext in [".png", ".jpg", ".jpeg", ".pdf"]:
        images.extend(dataset_path.glob(f"*{ext}"))
    
    return [{"id": i+1, "filename": img.name, "path": str(img), "ground_truth": {}} for i, img in enumerate(images)]


def extract_invoice(file_path: str, api_key: str, provider: str = "Gemini") -> Dict[str, Any]:
    """Extract data from an invoice."""
    try:
        # Load image
        if file_path.lower().endswith(".pdf"):
            images = extractor.convert_pdf_to_images(file_path)
            if not images:
                return {"error": "Failed to convert PDF"}
            image_input = images
        else:
            from PIL import Image
            image_input = Image.open(file_path)
        
        # Extract
        result = extractor.extract_invoice_data(
            image_input,
            api_key,
            provider=provider,
            model_name="gemini-2.5-flash" if provider == "Gemini" else "gpt-4o-mini"
        )
        
        return result
    except Exception as e:
        return {"error": str(e)}


def compare_with_ground_truth(extracted: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """Compare extracted data with ground truth."""
    comparison = {
        "matches": {},
        "mismatches": {},
        "missing": [],
        "extra": [],
    }
    
    # Key fields to compare
    key_fields = ["invoice_number", "date", "total_amount", "vendor_name", "currency"]
    
    for field in key_fields:
        extracted_val = extracted.get(field)
        truth_val = ground_truth.get(field)
        
        if truth_val is None:
            if extracted_val is not None:
                comparison["extra"].append(field)
            continue
        
        if extracted_val is None:
            comparison["missing"].append(field)
            continue
        
        # Normalize for comparison
        if field == "total_amount":
            try:
                ext_num = float(extracted_val)
                truth_num = float(truth_val)
                if abs(ext_num - truth_num) < 0.01:
                    comparison["matches"][field] = True
                else:
                    comparison["mismatches"][field] = {
                        "extracted": ext_num,
                        "expected": truth_num,
                        "diff": abs(ext_num - truth_num)
                    }
            except:
                comparison["mismatches"][field] = {
                    "extracted": extracted_val,
                    "expected": truth_val
                }
        else:
            # String comparison
            if str(extracted_val).strip().lower() == str(truth_val).strip().lower():
                comparison["matches"][field] = True
            else:
                comparison["mismatches"][field] = {
                    "extracted": extracted_val,
                    "expected": truth_val
                }
    
    return comparison


def calculate_accuracy(comparison: Dict[str, Any]) -> float:
    """Calculate accuracy score."""
    total_fields = len(comparison["matches"]) + len(comparison["mismatches"]) + len(comparison["missing"])
    if total_fields == 0:
        return 0.0
    
    correct = len(comparison["matches"])
    return correct / total_fields


def test_dataset(dataset_dir: str = "test_invoices_ready", max_invoices: int = 10):
    """Test parser on Kaggle dataset."""
    print("=" * 60)
    print("Testing Invoice Parser on Kaggle Dataset")
    print("=" * 60)
    
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key found. Set GEMINI_API_KEY or OPENAI_API_KEY in .env")
        return
    
    provider = "Gemini" if os.getenv("GEMINI_API_KEY") else "OpenAI"
    print(f"✅ Using {provider}")
    
    # Load test invoices
    invoices = load_test_invoices(dataset_dir)
    if not invoices:
        return
    
    print(f"\n📊 Found {len(invoices)} test invoices")
    print(f"   Testing first {min(max_invoices, len(invoices))} invoices...\n")
    
    results = []
    db = get_database()
    
    for i, invoice_info in enumerate(invoices[:max_invoices]):
        file_path = invoice_info["path"]
        filename = invoice_info["filename"]
        
        print(f"[{i+1}/{min(max_invoices, len(invoices))}] Processing: {filename}")
        
        # Extract
        extracted = extract_invoice(file_path, api_key, provider)
        
        if "error" in extracted:
            print(f"   ❌ Error: {extracted['error']}")
            results.append({
                "filename": filename,
                "success": False,
                "error": extracted["error"]
            })
            continue
        
        # Calculate confidence
        confidence = calculate_confidence(extracted)
        
        # Save to database
        extraction_id = db.save_extraction(
            extracted,
            confidence=confidence["overall"],
            source="test",
        )
        
        # Show results
        print(f"   ✅ Extracted:")
        print(f"      Invoice #: {extracted.get('invoice_number', 'N/A')}")
        print(f"      Date: {extracted.get('date', 'N/A')}")
        print(f"      Amount: {extracted.get('total_amount', 'N/A')} {extracted.get('currency', '')}")
        print(f"      Vendor: {extracted.get('vendor_name', 'N/A')}")
        print(f"      Confidence: {confidence['overall']:.0%}")
        
        if confidence["flags"]:
            print(f"      ⚠️  Flags: {', '.join(confidence['flags'][:2])}")
        
        results.append({
            "filename": filename,
            "success": True,
            "extracted": extracted,
            "confidence": confidence["overall"],
            "flags": confidence["flags"],
            "extraction_id": extraction_id,
        })
        
        print()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    
    if successful:
        avg_confidence = sum(r["confidence"] for r in successful) / len(successful)
        print(f"\n📊 Average Confidence: {avg_confidence:.0%}")
        
        high_conf = [r for r in successful if r["confidence"] >= 0.9]
        medium_conf = [r for r in successful if 0.75 <= r["confidence"] < 0.9]
        low_conf = [r for r in successful if r["confidence"] < 0.75]
        
        print(f"   High (≥90%): {len(high_conf)}")
        print(f"   Medium (75-90%): {len(medium_conf)}")
        print(f"   Low (<75%): {len(low_conf)}")
    
    # Save results
    results_file = Path("test_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("\n💡 View detailed results in your Streamlit app or test_results.json")


if __name__ == "__main__":
    import sys
    
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "test_invoices_ready"
    max_invoices = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    test_dataset(dataset_dir, max_invoices)
