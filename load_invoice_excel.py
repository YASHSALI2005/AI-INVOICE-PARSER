"""
Load and explore the Kaggle Invoice NER Dataset (Excel format).

The dataset zip contains: content@converted_invoice_dataset.xlsx
This script loads it, shows structure, and prepares data for testing.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    print("Install pandas: pip install pandas")
    exit(1)

try:
    import openpyxl
except ImportError:
    print("Install openpyxl for Excel support: pip install openpyxl")
    exit(1)


def find_excel_file(search_dir: str = ".") -> Optional[Path]:
    """Find the converted_invoice_dataset.xlsx file."""
    search_path = Path(search_dir)
    
    # Common locations after unzipping
    candidates = [
        search_path / "content@converted_invoice_dataset.xlsx",
        search_path / "converted_invoice_dataset.xlsx",
        search_path / "content" / "converted_invoice_dataset.xlsx",
    ]
    
    for p in candidates:
        if p.exists():
            return p
    
    # Search recursively
    for xlsx in search_path.rglob("*.xlsx"):
        if "invoice" in xlsx.name.lower() or "converted" in xlsx.name.lower():
            return xlsx
    
    return None


def load_excel(filepath: str) -> pd.DataFrame:
    """Load the Excel file into a DataFrame."""
    df = pd.read_excel(filepath, engine="openpyxl")
    return df


def analyze_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze Excel structure: columns, dtypes, sample values."""
    info = {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "sample": df.head(3).to_dict(orient="records"),
        "null_counts": df.isnull().sum().to_dict(),
    }
    return info


def parse_json_output(json_str: str) -> Dict[str, Any]:
    """Parse JSON string from Final_Output column."""
    if not json_str or pd.isna(json_str):
        return {}
    
    try:
        # Try to parse as JSON
        parsed = json.loads(json_str)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        # If not valid JSON, try to extract JSON-like content
        try:
            # Find JSON-like content between { }
            start = json_str.find("{")
            end = json_str.rfind("}") + 1
            if start >= 0 and end > start:
                json_content = json_str[start:end]
                return json.loads(json_content)
        except:
            pass
        return {}


def print_analysis(info: Dict[str, Any], df: pd.DataFrame):
    """Print analysis in a readable format."""
    print("\n" + "=" * 60)
    print("Dataset Structure")
    print("=" * 60)
    print(f"\nRows: {info['rows']}")
    print(f"\nColumns ({len(info['columns'])}):")
    for col in info["columns"]:
        nulls = info["null_counts"].get(col, 0)
        print(f"  - {col} (nulls: {nulls})")
    
    print("\nSample (first 3 rows):")
    for i, row in enumerate(info["sample"]):
        print(f"\n  Row {i+1}:")
        for k, v in row.items():
            val_str = str(v)[:100] + "..." if v is not None and len(str(v)) > 100 else v
            print(f"    {k}: {val_str}")
        
        # If Final_Output exists, try to parse and show structure
        if "Final_Output" in row or "final_output" in row:
            output_col = "Final_Output" if "Final_Output" in row else "final_output"
            output_val = row.get(output_col)
            if output_val:
                parsed = parse_json_output(str(output_val))
                if parsed:
                    print(f"    Parsed JSON keys: {list(parsed.keys())[:10]}")


def map_dataset_fields_to_schema(dataset_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map dataset JSON fields to our schema.
    
    Dataset uses keys like: TOTAL_AMOUNT, INVOICE_NUMBER, etc.
    Our schema uses: total_amount, invoice_number, etc.
    """
    mapping = {
        # Amount fields
        "TOTAL_AMOUNT": "total_amount",
        "AMOUNT": "total_amount",
        "GRAND_TOTAL": "total_amount",
        "DUE_AMOUNT": "due_amount",
        
        # Invoice number
        "INVOICE_NUMBER": "invoice_number",
        "INVOICE_NO": "invoice_number",
        "INVOICE_ID": "invoice_number",
        
        # Dates
        "DATE": "date",
        "INVOICE_DATE": "date",
        "ISSUE_DATE": "date",
        "DUE_DATE": "due_date",
        
        # Vendor/Company
        "VENDOR": "vendor_name",
        "VENDOR_NAME": "vendor_name",
        "COMPANY": "vendor_name",
        "SELLER": "vendor_name",
        "ISSUER": "vendor_name",
        "FROM": "vendor_name",
        
        # Currency
        "CURRENCY": "currency",
        "CURR": "currency",
        
        # Billing info
        "BILL_TO": "billing_address",
        "BILLED_TO": "billing_address",
        "CUSTOMER": "billing_address",
        
        # Other common fields
        "TAX": "tax",
        "SUBTOTAL": "subtotal",
        "DISCOUNT": "discount",
    }
    
    mapped = {}
    for dataset_key, our_key in mapping.items():
        if dataset_key in dataset_json:
            val = dataset_json[dataset_key]
            # Clean up values (remove $, convert to number if possible)
            if isinstance(val, str):
                val = val.strip()
                # Try to extract number from strings like "$1000"
                if our_key in ["total_amount", "due_amount", "tax", "subtotal", "discount"]:
                    import re
                    num_match = re.search(r'[\d,]+\.?\d*', val.replace(",", ""))
                    if num_match:
                        try:
                            val = float(num_match.group().replace(",", ""))
                        except:
                            pass
            mapped[our_key] = val
    
    # Also copy any unmapped fields
    for key, val in dataset_json.items():
        if key not in mapping:
            mapped[f"_{key.lower()}"] = val
    
    return mapped


def export_for_validation(df: pd.DataFrame, output_dir: str = "invoice_ground_truth"):
    """
    Export Excel rows as ground-truth JSON files for validation.
    
    Handles the Kaggle format:
    - "Input" column = Invoice text/template description
    - "Final_Output" column = JSON string with extracted fields
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Check if we have Input and Final_Output columns
    has_input = "Input" in df.columns or "input" in df.columns
    has_output = "Final_Output" in df.columns or "final_output" in df.columns or "Final Output" in df.columns
    
    records = []
    parsed_count = 0
    error_count = 0
    
    for idx, row in df.iterrows():
        record = {
            "_row": int(idx),
            "_input_text": None,
        }
        
        # Get input text if available
        input_col = None
        for col in ["Input", "input", "INPUT"]:
            if col in df.columns:
                input_col = col
                break
        
        if input_col:
            input_val = row.get(input_col)
            if not pd.isna(input_val):
                record["_input_text"] = str(input_val).strip()[:200]  # First 200 chars
        
        # Parse Final_Output JSON
        output_col = None
        for col in ["Final_Output", "final_output", "Final Output", "FINAL_OUTPUT"]:
            if col in df.columns:
                output_col = col
                break
        
        if output_col:
            output_val = row.get(output_col)
            if not pd.isna(output_val):
                dataset_json = parse_json_output(str(output_val))
                if dataset_json:
                    # Map to our schema
                    mapped = map_dataset_fields_to_schema(dataset_json)
                    record.update(mapped)
                    parsed_count += 1
                else:
                    error_count += 1
                    record["_parse_error"] = "Could not parse JSON"
        else:
            # Fallback: try to parse all columns as JSON or use direct mapping
            for col in df.columns:
                val = row.get(col)
                if not pd.isna(val):
                    try:
                        parsed = json.loads(str(val))
                        if isinstance(parsed, dict):
                            mapped = map_dataset_fields_to_schema(parsed)
                            record.update(mapped)
                            parsed_count += 1
                            break
                    except:
                        pass
        
        records.append(record)
    
    # Save as single JSON (list of ground truth)
    out_file = output_path / "ground_truth.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    
    print(f"\nExported {len(records)} records to {out_file}")
    print(f"  ✅ Successfully parsed JSON: {parsed_count}")
    if error_count > 0:
        print(f"  ⚠️  Parse errors: {error_count}")
    
    # Show sample of mapped fields
    if records and parsed_count > 0:
        sample = next((r for r in records if "total_amount" in r or "invoice_number" in r), None)
        if sample:
            print(f"\n  Sample mapped fields:")
            for key in ["invoice_number", "date", "total_amount", "vendor_name", "currency"]:
                if key in sample:
                    print(f"    {key}: {sample[key]}")
    
    return records, str(out_file)


def main():
    print("=" * 60)
    print("Invoice NER Dataset (Excel) Loader")
    print("=" * 60)
    
    # Find Excel file
    excel_path = find_excel_file(".")
    if not excel_path:
        # Check common download locations
        for d in ["test_invoices", "invoice_data", "downloads", "content"]:
            excel_path = find_excel_file(d)
            if excel_path:
                break
    
    if not excel_path:
        print("\nNo Excel file found.")
        print("Please place 'content@converted_invoice_dataset.xlsx' in this folder,")
        print("or run this script from the folder where you extracted the zip.")
        print("\nSearched: current dir, test_invoices/, invoice_data/, downloads/, content/")
        return
    
    print(f"\nFound: {excel_path}")
    
    # Load and analyze
    df = load_excel(str(excel_path))
    info = analyze_structure(df)
    print_analysis(info, df)
    
    # Export for validation
    print("\n" + "=" * 60)
    print("Export for validation")
    print("=" * 60)
    records, out_path = export_for_validation(df)
    
    print("\nDone. You can use ground_truth.json to compare parser output with expected values.")


if __name__ == "__main__":
    main()
