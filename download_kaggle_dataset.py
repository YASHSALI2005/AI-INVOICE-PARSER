"""
Script to download and use the Kaggle Invoice NER Dataset for testing.

Usage:
1. Install Kaggle API: pip install kaggle
2. Set up Kaggle credentials: https://www.kaggle.com/docs/api
3. Run: python download_kaggle_dataset.py
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any


def download_kaggle_dataset(dataset_name: str = "nikitpatel/invoice-ner-dataset", output_dir: str = "test_invoices"):
    """
    Download a Kaggle dataset.
    
    Requires:
    - kaggle package installed: pip install kaggle
    - Kaggle API credentials in ~/.kaggle/kaggle.json
    """
    try:
        import kaggle
    except ImportError:
        print("❌ Kaggle package not installed. Install with: pip install kaggle")
        print("   Then set up credentials: https://www.kaggle.com/docs/api")
        return False
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"📥 Downloading dataset: {dataset_name}")
    print(f"   Output directory: {output_path.absolute()}")
    
    try:
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(output_path),
            unzip=True,
            quiet=False
        )
        print(f"✅ Dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you've accepted the dataset terms on Kaggle")
        print("2. Check your Kaggle API credentials")
        print("3. Verify the dataset name is correct")
        return False


def analyze_dataset(dataset_dir: str = "test_invoices"):
    """Analyze the downloaded dataset structure."""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    print(f"\n📊 Analyzing dataset structure...")
    
    # Find all files
    files = list(dataset_path.rglob("*"))
    images = [f for f in files if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".pdf"]]
    json_files = [f for f in files if f.suffix == ".json"]
    txt_files = [f for f in files if f.suffix == ".txt"]
    
    print(f"\n📁 Files found:")
    print(f"   Images: {len(images)}")
    print(f"   JSON files: {len(json_files)}")
    print(f"   Text files: {len(txt_files)}")
    print(f"   Total files: {len(files)}")
    
    # Check for common dataset structures
    if json_files:
        print(f"\n📄 Sample JSON structure:")
        try:
            with open(json_files[0], "r", encoding="utf-8") as f:
                sample = json.load(f)
                print(f"   Keys: {list(sample.keys()) if isinstance(sample, dict) else 'List'}")
        except:
            pass
    
    # List directories
    dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    if dirs:
        print(f"\n📂 Directories:")
        for d in dirs:
            file_count = len(list(d.rglob("*")))
            print(f"   {d.name}/ ({file_count} files)")


def prepare_test_set(dataset_dir: str = "test_invoices", output_dir: str = "test_invoices_ready", max_files: int = 20):
    """
    Prepare a subset of the dataset for testing.
    Copies images and creates a simple index.
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all invoice images
    images = []
    for ext in [".png", ".jpg", ".jpeg", ".pdf"]:
        images.extend(dataset_path.rglob(f"*{ext}"))
    
    if not images:
        print(f"❌ No images found in {dataset_dir}")
        return
    
    print(f"\n🔄 Preparing test set...")
    print(f"   Found {len(images)} images")
    print(f"   Copying {min(max_files, len(images))} files...")
    
    test_files = []
    for i, img_path in enumerate(images[:max_files]):
        # Copy to output directory
        dest = output_path / img_path.name
        shutil.copy2(img_path, dest)
        test_files.append({
            "id": i + 1,
            "filename": img_path.name,
            "path": str(dest),
            "original_path": str(img_path),
        })
    
    # Create index file
    index_file = output_path / "index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(test_files, f, indent=2)
    
    print(f"✅ Test set prepared!")
    print(f"   Output: {output_path.absolute()}")
    print(f"   Index: {index_file}")
    print(f"\n💡 You can now test these invoices with your parser!")


def main():
    """Main function."""
    print("=" * 60)
    print("Kaggle Invoice Dataset Downloader")
    print("=" * 60)
    
    # Check if dataset already exists
    if Path("test_invoices").exists() and list(Path("test_invoices").rglob("*")):
        print("\n📁 Dataset already exists. Analyzing...")
        analyze_dataset("test_invoices")
        
        response = input("\n❓ Download fresh copy? (y/n): ").lower()
        if response != 'y':
            prepare_test_set()
            return
    
    # Download dataset
    print("\n" + "=" * 60)
    print("Step 1: Download Dataset")
    print("=" * 60)
    
    success = download_kaggle_dataset()
    
    if success:
        # Analyze
        print("\n" + "=" * 60)
        print("Step 2: Analyze Dataset")
        print("=" * 60)
        analyze_dataset()
        
        # Prepare test set
        print("\n" + "=" * 60)
        print("Step 3: Prepare Test Set")
        print("=" * 60)
        prepare_test_set()
        
        print("\n" + "=" * 60)
        print("✅ Done! You can now test invoices from test_invoices_ready/")
        print("=" * 60)
    else:
        print("\n💡 Manual download:")
        print("   1. Visit: https://www.kaggle.com/datasets/nikitpatel/invoice-ner-dataset")
        print("   2. Click 'Download' button")
        print("   3. Extract to 'test_invoices' folder")
        print("   4. Run this script again to prepare test set")


if __name__ == "__main__":
    main()
