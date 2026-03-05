"""
FastAPI backend for AI Invoice Parser.
- Auth: POST /auth/send-otp, POST /auth/verify-otp
- Extract: POST /api/extract (multipart: file, provider, document_type, use_hybrid)
- Data: GET /api/extractions/{id}, POST /api/extractions/{id}/corrections, GET /api/vendors, GET /api/stats
"""
import logging
from pathlib import Path

from dotenv import load_dotenv

# Load .env from this package directory so API keys are available regardless of cwd
load_dotenv(Path(__file__).resolve().parent / ".env")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.auth_router import router as auth_router
from api.extractions_router import router as extractions_router

logger = logging.getLogger("uvicorn.error")
db_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_ready
    try:
        from db import init_db
        init_db()
        db_ready = True
    except Exception as e:
        logger.warning("Database init failed (server will still start): %s", e)
    yield


app = FastAPI(
    title="AI Invoice Parser API",
    description="Upload invoices, extract data with LLM, store in PostgreSQL.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(extractions_router)


@app.get("/health")
def health():
    return {"status": "ok", "database": "ready" if db_ready else "not_ready"}
