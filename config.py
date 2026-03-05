"""Application config from environment."""
import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Settings loaded from env / .env."""

    # Database (PostgreSQL)
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:utpal@localhost:5432/retello_new?schema=public",
    )
    database_user: str = os.getenv("DATABASE_USER", "postgres")
    database_password: str = os.getenv("DATABASE_PASSWORD", "utpal")
    database_name: str = os.getenv("DATABASE_NAME", "retello_new")
    database_port: int = int(os.getenv("DATABASE_PORT", "5432"))
    database_schema: str = os.getenv("DATABASE_SCHEMA", "public")
    database_host: str = os.getenv("DATABASE_HOST", os.getenv("DATABSE_HOST", "localhost"))

    # SMS / OTP (auth)
    sms_api_url: str = os.getenv("SMS_API_URL", "")
    sms_api_username: str = os.getenv("SMS_API_USERNAME", "")
    sms_api_key: str = os.getenv("SMS_API_KEY", "")
    sms_sender_id: str = os.getenv("SMS_SENDER_ID", "")
    sms_route: str = os.getenv("SMS_ROUTE", "otp")

    # LLM API keys (for extraction)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    claude_api_key: str = os.getenv("CLAUDE_API_KEY", "")

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
