"""
Download and prepare the GitHub Invoice Dataset for testing.

Repo: https://github.com/mouadhamri/invoice_dataset.git
Contains: 9 invoice models × 100 invoices each = ~900 invoices with images + annotations
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List
import xml.etree.ElementTree as ET


def clone_repo(repo_url: str = "https://github.com/mouadhamri/invoice_dataset.git", target_dir: str = "invoice_dataset_github"):
    """Clone the GitHub repository."""
    target_path = Path(target_dir)
    
    if target_path.exists() and list(target_path.iterdir()):
        print(f"📁 Repository already exists at {target_path}")
        response = input("   Re-clone? (y/n): ").lower()
        if response != 'y':
            return str(target_path)
    
    print(f"📥 Cloning repository...")
    print(f"   URL: {repo_url}")
    print(f"   Target: {target_path.absolute()}")
    
    try:
        subprocess.run(
            ["git", "clone", repo_url, str(target_path)],
            check=True,
            capture_output=True
        )
        print(f"✅ Repository cloned successfully!")
        return str(target_path)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error cloning repository: {e}")
        print("\n💡 Manual download:")
        print(f"   1. Visit: {repo_url}")
        print(f"   2. Click 'Code' → 'Download ZIP'")
        print(f"   3. Extract to '{target_dir}' folder")
        return None
    except FileNotFoundError:
        print("❌ Git not found. Install Git or download manually.")
        print(f"   Visit: {repo_url}")
        return None


def analyze_dataset(dataset_dir: str = "invoice_dataset_github"):
    """Analyze the dataset structure."""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    print(f"\n📊 Analyzing dataset structure...")
    
    # Find all model folders
    model_folders = [d for d in dataset_path.iterdir() if d.is_dir() and "model" in d.name.lower()]
    model_folders.sort()
    
    print(f"\n📁 Found {len(model_folders)} invoice models:")
    for model_folder in model_folders:
        # Count files in each model
        images = list(model_folder.glob("*.png")) + list(model_folder.glob("*.jpg")) + list(model_folder.glob("*.pdf"))
        xml_files = list(model_folder.glob("*.xml"))
        json_files = list(model_folder.glob("*.json"))
        
        print(f"   {model_folder.name}:")
        print(f"      Images: {len(images)}")
        print(f"      XML annotations: {len(xml_files)}")
        print(f"      JSON files: {len(json_files)}")
    
    return model_folders


def parse_xml_annotation(xml_path: Path) -> Dict[str, Any]:
    """Parse XML annotation file to extract invoice fields."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        fields = {}
        
        # Common XML structures for invoice annotations
        # Try different possible structures
        for elem in root.iter():
            if elem.tag in ["field", "item", "key", "value"]:
                # Extract key-value pairs
                key = elem.get("name") or elem.get("key") or elem.get("label")
                value = elem.text or elem.get("value")
                if key and value:
                    fields[key.lower().replace(" ", "_")] = value.strip()
        
        # Also try direct text extraction
        if not fields:
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    tag = elem.tag.lower()
                    if tag not in ["root", "annotation", "invoice"]:
                        fields[tag] = elem.text.strip()
        
        return fields
    except Exception as e:
        print(f"   ⚠️  Error parsing {xml_path.name}: {e}")
        return {}


def prepare_test_set(dataset_dir: str = "invoice_dataset_github", output_dir: str = "test_invoices_github", max_per_model: int = 10):
    """
    Prepare a test set from the GitHub dataset.
    Copies images and creates ground truth JSON.
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    model_folders = [d for d in dataset_path.iterdir() if d.is_dir() and "model" in d.name.lower()]
    model_folders.sort()
    
    if not model_folders:
        print(f"❌ No model folders found in {dataset_dir}")
        return
    
    print(f"\n🔄 Preparing test set...")
    print(f"   Max {max_per_model} invoices per model")
    
    all_records = []
    total_copied = 0
    
    for model_folder in model_folders:
        print(f"\n   Processing {model_folder.name}...")
        
        # Find images
        images = []
        for ext in [".png", ".jpg", ".jpeg", ".pdf"]:
            images.extend(model_folder.glob(f"*{ext}"))
        
        # Limit per model
        images = images[:max_per_model]
        
        for img_path in images:
            # Copy image
            dest_img = output_path / f"{model_folder.name}_{img_path.name}"
            import shutil
            shutil.copy2(img_path, dest_img)
            
            # Find corresponding XML annotation
            xml_path = img_path.with_suffix(".xml")
            if not xml_path.exists():
                # Try other naming patterns
                base_name = img_path.stem
                for pattern in [f"{base_name}.xml", f"{base_name}_annotation.xml", f"{base_name}_gt.xml"]:
                    xml_path = model_folder / pattern
                    if xml_path.exists():
                        break
            
            # Parse XML if exists
            ground_truth = {}
            if xml_path.exists():
                ground_truth = parse_xml_annotation(xml_path)
            
            record = {
                "id": f"{model_folder.name}_{img_path.stem}",
                "model": model_folder.name,
                "image_path": str(dest_img),
                "original_path": str(img_path),
                "ground_truth": ground_truth,
            }
            
            all_records.append(record)
            total_copied += 1
    
    # Save index
    index_file = output_path / "index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
    
    print(f"\n✅ Test set prepared!")
    print(f"   Total invoices: {total_copied}")
    print(f"   Output: {output_path.absolute()}")
    print(f"   Index: {index_file}")
    
    return all_records


def main():
    print("=" * 60)
    print("GitHub Invoice Dataset Downloader")
    print("=" * 60)
    print("\nRepository: https://github.com/mouadhamri/invoice_dataset.git")
    print("Contains: ~900 invoices with images + annotations")
    
    # Clone repository
    print("\n" + "=" * 60)
    print("Step 1: Download Dataset")
    print("=" * 60)
    
    repo_path = clone_repo()
    
    if not repo_path:
        print("\n❌ Could not download repository.")
        print("   Please download manually and extract to 'invoice_dataset_github' folder")
        return
    
    # Analyze
    print("\n" + "=" * 60)
    print("Step 2: Analyze Dataset")
    print("=" * 60)
    
    model_folders = analyze_dataset(repo_path)
    
    if not model_folders:
        return
    
    # Prepare test set
    print("\n" + "=" * 60)
    print("Step 3: Prepare Test Set")
    print("=" * 60)
    
    max_per_model = 10
    try:
        user_input = input(f"\nHow many invoices per model? (default: {max_per_model}): ").strip()
        if user_input:
            max_per_model = int(user_input)
    except:
        pass
    
    records = prepare_test_set(repo_path, max_per_model=max_per_model)
    
    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)
    print(f"\n📁 Test invoices ready in: test_invoices_github/")
    print(f"📄 Index file: test_invoices_github/index.json")
    print(f"\n💡 You can now test your parser with:")
    print(f"   python test_kaggle_invoices.py test_invoices_github {len(records)}")


if __name__ == "__main__":
    main()
