import os
import argparse
import extractor
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Test AI Invoice Extraction CLI")
    parser.add_argument("file", help="Path to invoice file (PDF or Image)")
    args = parser.parse_args()

    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    # Simple logic for CLI: if GEMINI_API_KEY is present, use Gemini, else OpenAI
    if gemini_key:
        provider = "Gemini"
        api_key = gemini_key
    elif openai_key:
        provider = "OpenAI"
        api_key = openai_key
    else:
        print("Error: No API Key found.")
        return

    # Load the input file as image(s)
    file_path = args.file
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    images = None
    # Simple extension check: if PDF, convert to images, else open as image
    if file_path.lower().endswith(".pdf"):
        images = extractor.convert_pdf_to_images(open(file_path, "rb").read())
        if not images:
            print("Error: Failed to convert PDF to images.")
            return
        image_input = images  # pass list of pages
    else:
        try:
            image_input = Image.open(file_path)
        except Exception as e:
            print(f"Error opening image file: {e}")
            return

    print(f"Sending to {provider}...")
    data = extractor.extract_invoice_data(image_input, api_key, provider=provider)
    
    import json
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
