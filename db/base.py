"""SQLAlchemy declarative base (avoids circular import with models)."""
from sqlalchemy.orm import declarative_base

Base = declarative_base()
