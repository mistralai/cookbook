"""
Runnable version of data_extraction.ipynb
Run: python3 run_data_extraction.py
"""
import base64, json, os, sys
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field
from mistralai.client import Mistral
from mistralai.extra import response_format_from_pydantic_model

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("MISTRAL_API_KEY", "")
if not API_KEY:
    sys.exit("Set MISTRAL_API_KEY env var first.")

PDF_PATH = Path(__file__).parent / "mistral7b.pdf"
client = Mistral(api_key=API_KEY)

# ── Step 1: Encode PDF ────────────────────────────────────────────────────────
print("\n=== Step 1: Encoding PDF ===")
with open(PDF_PATH, "rb") as f:
    base64_pdf = base64.b64encode(f.read()).decode("utf-8")
print(f"PDF encoded ({len(base64_pdf)//1024} KB base64)")

# ── Step 2: Plain OCR (no annotations) ───────────────────────────────────────
print("\n=== Step 2: Plain OCR call (mistral-ocr-4-0) ===")
pdf_response = client.ocr.process(
    model="mistral-ocr-4-0",
    document={
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{base64_pdf}",
    },
    include_image_base64=False,
    extract_header=True,
    extract_footer=True,
    confidence_scores_granularity="word",
)

print(f"Pages returned: {len(pdf_response.pages)}")
for i, page in enumerate(pdf_response.pages[:3]):  # show first 3 pages
    words = page.confidence_scores.word_confidence_scores if page.confidence_scores else []
    avg_conf = (sum(w.confidence for w in words) / len(words) * 100) if words else 0
    low_conf = [w for w in words if w.confidence < 0.80]
    print(f"\n--- Page {i} ---")
    print(f"  Markdown length : {len(page.markdown)} chars")
    print(f"  Images detected : {len(page.images)}")
    print(f"  Words scored    : {len(words)}")
    print(f"  Avg confidence  : {avg_conf:.1f}%")
    print(f"  Low-conf words  : {len(low_conf)}")
    if low_conf:
        examples = [(w.text, round(w.confidence*100,1)) for w in low_conf[:5]]
        print(f"  Examples        : {examples}")
    print(f"  Markdown preview: {page.markdown[:300].strip()!r}")

# ── Step 3: OCR with Annotations ─────────────────────────────────────────────
print("\n\n=== Step 3: OCR with Annotations (mistral-ocr-latest) ===")

class ImageType(str, Enum):
    GRAPH = "graph"
    TEXT  = "text"
    TABLE = "table"
    IMAGE = "image"

class Image(BaseModel):
    image_type:  ImageType = Field(..., description="The type of the image. Must be one of 'graph', 'text', 'table' or 'image'.")
    description: str       = Field(..., description="A description of the image.")

class Document(BaseModel):
    language: str       = Field(..., description="The language of the document in ISO 639-1 code format (e.g., 'en', 'fr').")
    summary:  str       = Field(..., description="A summary of the document.")
    authors:  list[str] = Field(..., description="A list of authors who contributed to the document.")

ann_response = client.ocr.process(
    model="mistral-ocr-latest",
    pages=list(range(8)),
    document={
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{base64_pdf}",
    },
    bbox_annotation_format=response_format_from_pydantic_model(Image),
    document_annotation_format=response_format_from_pydantic_model(Document),
    include_image_base64=False,
)

print("\n--- Document Annotation ---")
print(ann_response.document_annotation)

print("\n--- BBox Annotations (all pages) ---")
for page in ann_response.pages:
    for img in page.images:
        print(f"\n  [{img.id}] @ ({img.top_left_x:.0f},{img.top_left_y:.0f}) → ({img.bottom_right_x:.0f},{img.bottom_right_y:.0f})")
        print(f"  Annotation: {img.image_annotation}")

# ── Step 4: Save full JSON response ──────────────────────────────────────────
out = Path(__file__).parent / "ocr_output.json"
with open(out, "w") as f:
    json.dump(json.loads(ann_response.model_dump_json()), f, indent=2)
print(f"\n\nFull annotated response saved → {out}")
