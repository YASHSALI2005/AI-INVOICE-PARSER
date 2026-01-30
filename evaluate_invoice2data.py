import os
import json
import logging
from invoice2data import extract_data
from invoice2data.extract.loader import read_templates
from invoice2data.input import pdftotext

# Set log level to INFO
logging.basicConfig(level=logging.INFO)

test_dir = r"c:\Users\yashs\Downloads\test\invoice2data_repo\tests\compare"
template_dir = r"c:\Users\yashs\Downloads\test\invoice2data_repo\src\invoice2data\extract\templates"

# Add Poppler to PATH
poppler_path = r"C:\Users\yashs\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin"
os.environ["PATH"] += os.pathsep + poppler_path

# Load templates from the repo
print(f"Loading templates from {template_dir}...")
templates = read_templates(template_dir)
print(f"Loaded {len(templates)} templates.")

files = [f for f in os.listdir(test_dir) if f.endswith('.pdf')]

print(f"Found {len(files)} PDF files to test in {test_dir}")

for file in files:
    file_path = os.path.join(test_dir, file)
    print(f"\nProcessing {file}...")
    try:
        # Pass the module directly
        result = extract_data(file_path, templates=templates, input_module=pdftotext)
        
        if result:
            print(f"  [SUCCESS] Extracted data.")
            # Print a few key fields
            for key in ['issuer', 'date', 'amount', 'invoice_number', 'currency']:
                if key in result:
                    print(f"    {key}: {result[key]}")
            
            # Compare with expected json if exists
            json_file = file.replace('.pdf', '.json')
            json_path = os.path.join(test_dir, json_file)
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    expected = json.load(f)
                
                # Check amount
                if 'amount' in expected and 'amount' in result:
                    try:
                        exp_amt = float(expected['amount'])
                        res_amt = float(result['amount'])
                        if abs(exp_amt - res_amt) < 0.01:
                            print(f"    [MATCH] Amount matches: {res_amt}")
                        else:
                            print(f"    [MISMATCH] Amount expected: {exp_amt}, got: {res_amt}")
                    except ValueError:
                         print(f"    [INFO] Amount comparison skipped due to type mismatch: {expected['amount']} vs {result['amount']}")

                # Check date if simple string match or comparable
                if 'date' in expected and 'date' in result:
                    # Dates in result might be datetime objects or tuples depending on version
                    print(f"    [DEBUG] Date result raw: {result['date']} (Type: {type(result['date'])})")
            else:
                print("  [INFO] No corresponding JSON file for verification.")

        else:
            print(f"  [FAILED] No data extracted (no template matched?).")
            
    except Exception as e:
        print(f"  [ERROR] Exception processing {file}: {e}")
