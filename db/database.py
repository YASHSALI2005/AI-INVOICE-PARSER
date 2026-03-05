"""PostgreSQL connection and session."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from config import get_settings
from .base import Base

# Import models so they register with Base.metadata (before init_db)
from . import models  # noqa: F401

_settings = get_settings()
# psycopg2 does not support "schema" in DSN; tables use __table_args__ = {"schema": "public"}
DATABASE_URL = _settings.database_url
if "?schema=" in DATABASE_URL or "&schema=" in DATABASE_URL:
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
    parsed = urlparse(DATABASE_URL)
    qs = parse_qs(parsed.query)
    qs.pop("schema", None)
    new_query = urlencode(qs, doseq=True)
    DATABASE_URL = urlunparse(parsed._replace(query=new_query))

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """Dependency that yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)
