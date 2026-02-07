
import os
import sys
import json
from dotenv import load_dotenv

# Load env for API Key
load_dotenv()
api_key = os.getenv("CLAUDE_API_KEY")

if not api_key:
    # Fallback/Check
    print("Error: CLAUDE_API_KEY not found in .env")
    sys.exit(1)

import verified_extractor

def main():
    # Find a sample file (PDF or Image)
    # Check invoice_data for PDFs first
    sample_dir = "invoice_data"
    files = []
    if os.path.exists(sample_dir):
        files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.lower().endswith(".pdf")]
    
    # If no PDFs, check for images in the dataset folder
    if not files:
        print(f"No PDFs in {sample_dir}, checking secondary image source...")
        secondary_dir = r"invoice_dataset_github\invoice_dataset_model_1\images"
        if os.path.exists(secondary_dir):
            files = [os.path.join(secondary_dir, f) for f in os.listdir(secondary_dir) if f.lower().endswith((".jpg", ".png"))]

    if not files:
        print("No PDF or Image files found to test.")
        return

    sample_file = files[0]
    print(f"Testing with: {sample_file}")

    result = verified_extractor.extract_with_verification(
        file_input=sample_file,
        api_key=api_key,
        provider="Claude",
        model_name="claude-3-haiku-20240307"
    )

    if "error" in result:
        print("\nWorkflow Failed:", result["error"])
        if "message" in result:
            print("Reason:", result["message"])
    else:
        print("\n=== Initial Extraction (Snippet) ===")
        print(json.dumps(result["initial_extraction"]["summary"], indent=2))
        
        print("\n=== Final Verified (Snippet) ===")
        print(json.dumps(result["final_verified_extraction"]["summary"], indent=2))
        
        print("\nFull result saved to verified_result.json")
        with open("verified_result.json", "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
