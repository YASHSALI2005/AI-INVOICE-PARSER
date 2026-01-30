"""
Invoice Extraction Database - Stores extractions for learning and improvement.

Features:
- Store successful extractions with vendor fingerprint
- Track user corrections
- Retrieve similar past extractions
- Support template generation
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class InvoiceDatabase:
    """SQLite-free JSON-based database for invoice extractions."""
    
    def __init__(self, db_dir: str = "invoice_data"):
        self.db_dir = Path(db_dir)
        self.extractions_dir = self.db_dir / "extractions"
        self.templates_dir = self.db_dir / "templates"
        self.corrections_dir = self.db_dir / "corrections"
        self.vendors_file = self.db_dir / "vendors.json"
        
        # Create directories
        self.extractions_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.corrections_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize vendors index
        self.vendors = self._load_vendors()
    
    def _load_vendors(self) -> Dict[str, Any]:
        """Load vendors index."""
        if self.vendors_file.exists():
            with open(self.vendors_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"vendors": {}, "fingerprints": {}}
    
    def _save_vendors(self):
        """Save vendors index."""
        with open(self.vendors_file, "w", encoding="utf-8") as f:
            json.dump(self.vendors, f, indent=2)
    
    def _generate_vendor_fingerprint(self, vendor_name: str) -> str:
        """Generate a fingerprint for vendor matching."""
        if not vendor_name:
            return "unknown"
        # Normalize vendor name
        normalized = vendor_name.lower().strip()
        # Remove common suffixes
        for suffix in [", inc.", ", inc", " inc.", " inc", ", llc", " llc", ", ltd", " ltd", " limited", " corporation", " corp", " corp."]:
            normalized = normalized.replace(suffix, "")
        return normalized.strip()
    
    def _generate_extraction_id(self, data: Dict[str, Any]) -> str:
        """Generate unique ID for extraction."""
        content = json.dumps(data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def save_extraction(
        self,
        extracted_data: Dict[str, Any],
        confidence: float = 1.0,
        source: str = "llm",
        image_hash: Optional[str] = None,
    ) -> str:
        """
        Save an extraction to the database.
        
        Returns the extraction ID.
        """
        extraction_id = self._generate_extraction_id(extracted_data)
        vendor_name = extracted_data.get("vendor_name", "Unknown")
        vendor_fp = self._generate_vendor_fingerprint(vendor_name)
        
        record = {
            "id": extraction_id,
            "timestamp": datetime.now().isoformat(),
            "vendor_name": vendor_name,
            "vendor_fingerprint": vendor_fp,
            "data": extracted_data,
            "confidence": confidence,
            "source": source,  # "llm", "template", "hybrid"
            "image_hash": image_hash,
            "corrections": [],
            "verified": False,
        }
        
        # Save extraction
        filepath = self.extractions_dir / f"{extraction_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        
        # Update vendors index
        if vendor_fp not in self.vendors["vendors"]:
            self.vendors["vendors"][vendor_fp] = {
                "name": vendor_name,
                "extraction_count": 0,
                "template_exists": False,
                "first_seen": datetime.now().isoformat(),
            }
        self.vendors["vendors"][vendor_fp]["extraction_count"] += 1
        self.vendors["vendors"][vendor_fp]["last_seen"] = datetime.now().isoformat()
        
        # Add fingerprint mapping
        if vendor_fp not in self.vendors["fingerprints"]:
            self.vendors["fingerprints"][vendor_fp] = []
        if extraction_id not in self.vendors["fingerprints"][vendor_fp]:
            self.vendors["fingerprints"][vendor_fp].append(extraction_id)
        
        self._save_vendors()
        
        return extraction_id
    
    def get_extraction(self, extraction_id: str) -> Optional[Dict[str, Any]]:
        """Get an extraction by ID."""
        filepath = self.extractions_dir / f"{extraction_id}.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    
    def save_correction(
        self,
        extraction_id: str,
        original_data: Dict[str, Any],
        corrected_data: Dict[str, Any],
        corrected_fields: List[str],
    ) -> bool:
        """
        Save a user correction for an extraction.
        """
        extraction = self.get_extraction(extraction_id)
        if not extraction:
            return False
        
        correction = {
            "timestamp": datetime.now().isoformat(),
            "original": original_data,
            "corrected": corrected_data,
            "fields_changed": corrected_fields,
        }
        
        # Add to extraction record
        extraction["corrections"].append(correction)
        extraction["verified"] = True
        
        filepath = self.extractions_dir / f"{extraction_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(extraction, f, indent=2)
        
        # Also save to corrections directory for easy access
        correction_file = self.corrections_dir / f"{extraction_id}_{len(extraction['corrections'])}.json"
        with open(correction_file, "w", encoding="utf-8") as f:
            json.dump(correction, f, indent=2)
        
        return True
    
    def get_vendor_extractions(self, vendor_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get past extractions for a vendor."""
        vendor_fp = self._generate_vendor_fingerprint(vendor_name)
        extraction_ids = self.vendors.get("fingerprints", {}).get(vendor_fp, [])
        
        extractions = []
        for eid in extraction_ids[-limit:]:
            ext = self.get_extraction(eid)
            if ext:
                extractions.append(ext)
        
        return extractions
    
    def has_template(self, vendor_name: str) -> bool:
        """Check if a template exists for this vendor."""
        vendor_fp = self._generate_vendor_fingerprint(vendor_name)
        template_file = self.templates_dir / f"{vendor_fp}.yml"
        return template_file.exists()
    
    def save_template(self, vendor_name: str, template_content: str) -> str:
        """Save a generated template."""
        vendor_fp = self._generate_vendor_fingerprint(vendor_name)
        template_file = self.templates_dir / f"{vendor_fp}.yml"
        
        with open(template_file, "w", encoding="utf-8") as f:
            f.write(template_content)
        
        # Update vendors index
        if vendor_fp in self.vendors["vendors"]:
            self.vendors["vendors"][vendor_fp]["template_exists"] = True
            self._save_vendors()
        
        return str(template_file)
    
    def get_template(self, vendor_name: str) -> Optional[str]:
        """Get template content for a vendor."""
        vendor_fp = self._generate_vendor_fingerprint(vendor_name)
        template_file = self.templates_dir / f"{vendor_fp}.yml"
        
        if template_file.exists():
            with open(template_file, "r", encoding="utf-8") as f:
                return f.read()
        return None
    
    def get_all_vendors(self) -> List[Dict[str, Any]]:
        """Get list of all known vendors."""
        vendors = []
        for fp, data in self.vendors.get("vendors", {}).items():
            vendors.append({
                "fingerprint": fp,
                **data
            })
        return sorted(vendors, key=lambda x: x.get("extraction_count", 0), reverse=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        extraction_count = len(list(self.extractions_dir.glob("*.json")))
        template_count = len(list(self.templates_dir.glob("*.yml")))
        correction_count = len(list(self.corrections_dir.glob("*.json")))
        vendor_count = len(self.vendors.get("vendors", {}))
        
        return {
            "total_extractions": extraction_count,
            "total_templates": template_count,
            "total_corrections": correction_count,
            "total_vendors": vendor_count,
        }


# Global database instance
_db_instance: Optional[InvoiceDatabase] = None


def get_database() -> InvoiceDatabase:
    """Get or create the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = InvoiceDatabase()
    return _db_instance
