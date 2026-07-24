"""
Configuration for Mistral Document AI OCR on Azure.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Azure-hosted Mistral OCR endpoint ──────────────────────────────────────────
MISTRAL_OCR_ENDPOINT = os.getenv("MISTRAL_OCR_ENDPOINT", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# The Azure AI model inference base URL (without the trailing /ocr)
MISTRAL_SERVER_URL = MISTRAL_OCR_ENDPOINT.rsplit("/ocr", 1)[0]

# Model name used for the Mistral SDK call
MISTRAL_OCR_MODEL = os.getenv("MISTRAL_OCR_MODEL", "mistral-document-ai-2505")

# ── Mistral chat model (used for verification / structured extraction) ─────────
MISTRAL_CHAT_MODEL = os.getenv("MISTRAL_CHAT_MODEL", "mistral-large-latest")

# ── Chat model (Azure OpenAI for document Q&A) ───────────────────────────────
CHAT_AZURE_ENDPOINT = os.getenv("CHAT_AZURE_ENDPOINT", "")
CHAT_API_KEY = os.getenv("CHAT_API_KEY", "")
CHAT_API_VERSION = os.getenv("CHAT_API_VERSION", "2025-01-01-preview")
CHAT_MODEL = os.getenv("CHAT_MODEL", "Mistral-Large-3")

# ── Application settings ──────────────────────────────────────────────────────
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Maximum pages to process (0 = all)
MAX_PAGES = int(os.getenv("MAX_PAGES", "0"))
