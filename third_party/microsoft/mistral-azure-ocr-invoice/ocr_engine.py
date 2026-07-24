"""
Mistral Document AI OCR engine.

Supports:
  - Azure-hosted Mistral OCR via REST API
  - Fallback to official Mistral Python SDK
  - Local file (base64-encoded) and URL-based documents

Returns raw markdown / HTML tables per page.
"""

import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

import config

# Azure AD token cache
_azure_token_cache: Dict[str, Any] = {}

logger = logging.getLogger(__name__)

# Suppress noisy Azure identity warnings when key auth is used
logging.getLogger("azure.identity").setLevel(logging.ERROR)
logging.getLogger("azure.core").setLevel(logging.ERROR)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class OCRImage:
    id: str
    top_left_x: float = 0
    top_left_y: float = 0
    bottom_right_x: float = 0
    bottom_right_y: float = 0
    image_base64: Optional[str] = None


@dataclass
class OCRPage:
    index: int
    markdown: str
    html_tables: List[str] = field(default_factory=list)
    images: List[OCRImage] = field(default_factory=list)
    dimensions: Optional[Dict[str, float]] = None


@dataclass
class OCRResult:
    pages: List[OCRPage]
    raw_response: Dict[str, Any]
    model: str = ""
    usage: Optional[Dict[str, int]] = None
    processing_time_s: float = 0.0

    @property
    def full_markdown(self) -> str:
        return "\n\n---\n\n".join(p.markdown for p in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_pdf(pdf_path: str) -> Optional[str]:
    """Encode a PDF file to base64."""
    try:
        return base64.b64encode(Path(pdf_path).read_bytes()).decode("utf-8")
    except Exception as exc:
        logger.error("Failed to encode PDF %s: %s", pdf_path, exc)
        return None


def _extract_html_tables(md: str) -> List[str]:
    """Pull <table>…</table> blocks out of markdown/HTML content."""
    return re.findall(r"(<table[\s\S]*?</table>)", md, re.IGNORECASE)


# ── Azure AD Token helper ─────────────────────────────────────────────────────

def _get_azure_ad_token() -> Optional[str]:
    """
    Obtain an Azure AD bearer token for Cognitive Services.
    Tries: AzureCliCredential → DefaultAzureCredential.
    Returns None if no credential is available.
    """
    import time as _time

    # Return cached token if still valid (with 5-min buffer)
    cached = _azure_token_cache.get("token")
    expires = _azure_token_cache.get("expires_on", 0)
    if cached and _time.time() < expires - 300:
        return cached

    scope = "https://cognitiveservices.azure.com/.default"

    # Try AzureCliCredential first (most common in dev)
    try:
        from azure.identity import AzureCliCredential  # type: ignore
        cred = AzureCliCredential()
        tok = cred.get_token(scope)
        _azure_token_cache["token"] = tok.token
        _azure_token_cache["expires_on"] = tok.expires_on
        logger.info("Obtained Azure AD token via AzureCliCredential")
        return tok.token
    except Exception as e:
        logger.debug("AzureCliCredential failed: %s", e)

    # Try DefaultAzureCredential as fallback
    try:
        from azure.identity import DefaultAzureCredential  # type: ignore
        cred = DefaultAzureCredential()
        tok = cred.get_token(scope)
        _azure_token_cache["token"] = tok.token
        _azure_token_cache["expires_on"] = tok.expires_on
        logger.info("Obtained Azure AD token via DefaultAzureCredential")
        return tok.token
    except Exception as e:
        logger.debug("DefaultAzureCredential failed: %s", e)

    return None


# ── REST-based OCR call (Azure AI Inference) ──────────────────────────────────

def _ocr_via_rest(
    document_base64: Optional[str] = None,
    document_url: Optional[str] = None,
    include_image_base64: bool = False,
) -> Dict[str, Any]:
    """Call the Mistral OCR endpoint via plain REST / Azure AI Inference."""

    if document_base64:
        doc_payload = {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{document_base64}",
        }
    elif document_url:
        # This Azure deployment only supports base64; download and convert
        logger.info("Downloading remote PDF to convert to base64 (URL documents not supported by this deployment)")
        try:
            dl = httpx.get(document_url, timeout=60, follow_redirects=True)
            dl.raise_for_status()
            b64 = base64.b64encode(dl.content).decode("utf-8")
            doc_payload = {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{b64}",
            }
        except Exception as exc:
            raise ValueError(f"Failed to download PDF from URL: {exc}")
    else:
        raise ValueError("Provide either document_base64 or document_url")

    body: Dict[str, Any] = {
        "model": config.MISTRAL_OCR_MODEL,
        "document": doc_payload,
        "include_image_base64": include_image_base64,
    }

    # ── Build auth strategies ─────────────────────────────────────────────────
    auth_strategies = []

    # Strategy 1: Azure AD token (required when key-based auth is disabled)
    ad_token = _get_azure_ad_token()
    if ad_token:
        auth_strategies.append({
            "name": "AzureAD",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {ad_token}",
                "Accept": "application/json",
            },
        })

    # Strategy 2: API key (api-key header — Azure style)
    if config.MISTRAL_API_KEY:
        auth_strategies.append({
            "name": "ApiKey",
            "headers": {
                "Content-Type": "application/json",
                "api-key": config.MISTRAL_API_KEY,
                "Accept": "application/json",
            },
        })
        # Strategy 3: Bearer key (Mistral platform style)
        auth_strategies.append({
            "name": "BearerKey",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.MISTRAL_API_KEY}",
                "Accept": "application/json",
            },
        })

    if not auth_strategies:
        raise RuntimeError(
            "No authentication available. Either:\n"
            "  1. Run 'az login' to authenticate with Azure AD, or\n"
            "  2. Set MISTRAL_API_KEY env variable (if key-auth is enabled on the resource)"
        )

    # ── Try each auth strategy against each endpoint URL ──────────────────────
    urls_to_try = [
        config.MISTRAL_OCR_ENDPOINT,
        f"{config.MISTRAL_SERVER_URL}/v1/ocr",
    ]

    last_error = None
    for strategy in auth_strategies:
        for url in urls_to_try:
            try:
                logger.info("Trying OCR: %s @ %s", strategy["name"], url)
                resp = httpx.post(url, json=body, headers=strategy["headers"], timeout=120)
                if resp.status_code == 200:
                    logger.info("OCR success via %s @ %s", strategy["name"], url)
                    return resp.json()
                logger.warning("HTTP %s from %s [%s]: %s",
                               resp.status_code, url, strategy["name"], resp.text[:300])
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            except Exception as exc:
                logger.warning("Request to %s [%s] failed: %s", url, strategy["name"], exc)
                last_error = str(exc)

    # Provide actionable error message
    error_msg = f"All OCR endpoints/auth methods failed. Last error: {last_error}"
    if "AuthenticationTypeDisabled" in (last_error or ""):
        error_msg += (
            "\n\n*** Key-based auth is disabled on this Azure resource. ***\n"
            "Please run 'az login --use-device-code' in the terminal first,\n"
            "then retry the analysis."
        )
    raise RuntimeError(error_msg)


# ── SDK-based OCR call ────────────────────────────────────────────────────────

def _ocr_via_sdk(
    document_base64: Optional[str] = None,
    document_url: Optional[str] = None,
    include_image_base64: bool = False,
) -> Dict[str, Any]:
    """Call Mistral OCR using the official Python SDK."""
    try:
        from mistralai import Mistral  # type: ignore
    except ImportError:
        raise ImportError("Install mistralai: pip install mistralai")

    client = Mistral(
        api_key=config.MISTRAL_API_KEY,
        server_url=config.MISTRAL_SERVER_URL,
    )

    if document_base64:
        from mistralai.models import DocumentURLChunk  # type: ignore
        doc = DocumentURLChunk(
            document_url=f"data:application/pdf;base64,{document_base64}",
        )
    elif document_url:
        from mistralai.models import DocumentURLChunk  # type: ignore
        doc = DocumentURLChunk(document_url=document_url)
    else:
        raise ValueError("Provide either document_base64 or document_url")

    response = client.ocr.process(
        model=config.MISTRAL_OCR_MODEL,
        document=doc,
        include_image_base64=include_image_base64,
    )

    return response.model_dump() if hasattr(response, "model_dump") else json.loads(json.dumps(response, default=str))


# ── Public API ────────────────────────────────────────────────────────────────

def process_pdf(
    pdf_path: Optional[str] = None,
    pdf_url: Optional[str] = None,
    include_images: bool = False,
    prefer_sdk: bool = False,
) -> OCRResult:
    """
    Process a PDF through Mistral Document AI OCR.

    Args:
        pdf_path: Local path to a PDF file.
        pdf_url:  URL of a remote PDF.
        include_images: Whether to return base64-encoded images.
        prefer_sdk: Try the Mistral SDK first (else REST first).

    Returns:
        OCRResult with pages, markdown, tables, images.
    """
    doc_b64 = encode_pdf(pdf_path) if pdf_path else None
    if pdf_path and doc_b64 is None:
        raise FileNotFoundError(f"Cannot read PDF: {pdf_path}")

    t0 = time.time()
    methods = [_ocr_via_sdk, _ocr_via_rest] if prefer_sdk else [_ocr_via_rest, _ocr_via_sdk]

    raw: Dict[str, Any] = {}
    for method in methods:
        try:
            raw = method(
                document_base64=doc_b64,
                document_url=pdf_url,
                include_image_base64=include_images,
            )
            break
        except Exception as exc:
            logger.warning("%s failed: %s", method.__name__, exc)
            continue
    else:
        raise RuntimeError("All OCR methods failed")

    elapsed = time.time() - t0

    # Parse response into OCRResult
    pages: List[OCRPage] = []
    raw_pages = raw.get("pages") or raw.get("data", {}).get("pages", [])
    for idx, rp in enumerate(raw_pages):
        md = rp.get("markdown", "") or rp.get("content", "")
        html_tables = _extract_html_tables(md)

        images = []
        for img_data in rp.get("images", []):
            images.append(OCRImage(
                id=img_data.get("id", f"img_{idx}"),
                top_left_x=img_data.get("top_left_x", 0),
                top_left_y=img_data.get("top_left_y", 0),
                bottom_right_x=img_data.get("bottom_right_x", 0),
                bottom_right_y=img_data.get("bottom_right_y", 0),
                image_base64=img_data.get("image_base64"),
            ))

        pages.append(OCRPage(
            index=idx,
            markdown=md,
            html_tables=html_tables,
            images=images,
            dimensions=rp.get("dimensions"),
        ))

    return OCRResult(
        pages=pages,
        raw_response=raw,
        model=raw.get("model", config.MISTRAL_OCR_MODEL),
        usage=raw.get("usage"),
        processing_time_s=round(elapsed, 2),
    )
