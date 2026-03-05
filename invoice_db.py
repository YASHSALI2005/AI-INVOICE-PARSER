"""
Invoice Extraction Database - Stores extractions for learning and improvement.
Backed by PostgreSQL (see DATABASE_URL in .env).
"""

from typing import Optional

from db.repository import PostgresInvoiceRepository

_db_instance: Optional[PostgresInvoiceRepository] = None


def get_database() -> PostgresInvoiceRepository:
    """Get or create the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = PostgresInvoiceRepository()
    return _db_instance
