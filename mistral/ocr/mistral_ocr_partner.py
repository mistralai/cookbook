"""
Mistral OCR — Production-Ready Partner Script
==============================================
Demonstrates four patterns not covered in Mistral's public docs:

  1. Confidence quality gate   — flag pages below threshold for human review
  2. Long-doc chunking         — handle docs > 8 pages with document_annotation
  3. Annotation context bridge — attach surrounding markdown text to each bbox
                                 annotation so misclassifications are catchable
  4. Safe response parsing     — document_annotation and image_annotation are
                                 JSON strings, not Pydantic objects; parse them

Usage:
    export MISTRAL_API_KEY="..."
    python3 mistral_ocr_partner.py --pdf mistral7b.pdf
    python3 mistral_ocr_partner.py --url https://example.com/doc.pdf
    python3 mistral_ocr_partner.py --pdf invoice.pdf --confidence-threshold 0.90
"""

import argparse, base64, json, os, sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from mistralai.client import Mistral
from mistralai.extra import response_format_from_pydantic_model

# ── Annotation schemas ────────────────────────────────────────────────────────
# Field descriptions are the prompt — be specific.

class ImageType(str, Enum):
    GRAPH = "graph"
    TABLE = "table"
    TEXT  = "text"
    IMAGE = "image"

class BBoxAnnotation(BaseModel):
    image_type:  ImageType = Field(..., description="Type of visual element: 'graph' for charts/plots, 'table' for tabular data, 'text' for text-heavy figures, 'image' for photographs or illustrations.")
    description: str       = Field(..., description="One-paragraph description of the content. For graphs, state the axes, series, and the main comparative finding (e.g. 'Model A outperforms Model B in category X'). For tables, describe column headers and the key data relationship.")

class DocumentAnnotation(BaseModel):
    language: str       = Field(..., description="Primary language of the document as ISO 639-1 code (e.g. 'en', 'fr').")
    summary:  str       = Field(..., description="3–5 sentence technical summary covering the document's purpose, key findings, and methodology.")
    authors:  list[str] = Field(default_factory=list, description="Full names of all authors or contributors listed in the document.")
    doc_type: str       = Field(..., description="Document category: one of 'research_paper', 'invoice', 'contract', 'report', 'form', 'presentation', 'other'.")


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class PageQuality:
    page_index:    int
    avg_confidence: float
    low_conf_words: list[tuple[str, float]]  # (word, confidence)
    flagged:       bool

@dataclass
class BBoxResult:
    page_index:  int
    image_id:    str
    coords:      tuple[float, float, float, float]  # x1, y1, x2, y2
    annotation:  dict
    context_text: str   # surrounding markdown — for human validation

@dataclass
class OCRReport:
    total_pages:        int
    flagged_pages:      list[PageQuality]
    document_annotation: dict
    bbox_results:       list[BBoxResult]
    raw_markdown:       dict[int, str]  # page_index → markdown


# ── Core functions ────────────────────────────────────────────────────────────

def load_document(pdf_path: Optional[str], url: Optional[str]) -> tuple[str, str]:
    """Return (document_url, label) ready for the API."""
    if url:
        return url, url
    path = Path(pdf_path)
    if not path.exists():
        sys.exit(f"File not found: {path}")
    b64 = base64.b64encode(path.read_bytes()).decode()
    ext = path.suffix.lower().lstrip(".")
    mime = {"pdf": "application/pdf", "png": "image/png",
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            }.get(ext, "application/pdf")
    return f"data:{mime};base64,{b64}", str(path.name)


def run_plain_ocr(client: Mistral, doc_url: str, confidence_threshold: float) -> tuple[list[PageQuality], dict[int, str], list]:
    """
    Plain OCR pass — returns quality assessment per page.
    Confidence threshold (0.0–1.0): pages below this avg are flagged for review.
    """
    print("\n[1/3] Running plain OCR with confidence scoring...")
    response = client.ocr.process(
        model="mistral-ocr-4-0",
        document={"type": "document_url", "document_url": doc_url},
        include_image_base64=False,
        extract_header=True,
        extract_footer=True,
        confidence_scores_granularity="word",
    )

    quality_results = []
    markdown_by_page = {}

    for page in response.pages:
        words = []
        if page.confidence_scores and page.confidence_scores.word_confidence_scores:
            words = page.confidence_scores.word_confidence_scores

        # Filter out OCR-generated structural tokens (###, ---|---) from quality calc
        content_words = [w for w in words if not set(w.text.strip()).issubset(set("#|-_ \t"))]
        avg = (sum(w.confidence for w in content_words) / len(content_words)) if content_words else 1.0
        low_conf = [(w.text.strip(), round(w.confidence * 100, 1))
                    for w in content_words if w.confidence < 0.80]

        pq = PageQuality(
            page_index=page.index,
            avg_confidence=avg,
            low_conf_words=low_conf[:10],
            flagged=avg < confidence_threshold,
        )
        quality_results.append(pq)
        markdown_by_page[page.index] = page.markdown

    flagged = [p for p in quality_results if p.flagged]
    print(f"    Pages: {len(response.pages)} | Flagged for review: {len(flagged)}")
    if flagged:
        for p in flagged:
            print(f"    ⚠  Page {p.page_index}: avg confidence {p.avg_confidence*100:.1f}% — low-conf words: {p.low_conf_words[:3]}")

    return quality_results, markdown_by_page, response.pages


def extract_bbox_context(markdown: str, image_id: str, window: int = 200) -> str:
    """
    Pull the text surrounding an image placeholder in the page markdown.
    This gives human reviewers context to spot annotation errors
    (e.g. 'A outperforms B' when the text says the opposite).
    """
    placeholder = f"![{image_id}]({image_id})"
    idx = markdown.find(placeholder)
    if idx == -1:
        return ""
    start = max(0, idx - window)
    end   = min(len(markdown), idx + len(placeholder) + window)
    before = markdown[start:idx].strip().replace("\n", " ")
    after  = markdown[idx + len(placeholder):end].strip().replace("\n", " ")
    return f"...{before} [IMAGE] {after}...".strip()


def run_annotated_ocr_chunked(client: Mistral, doc_url: str, total_pages: int) -> tuple[dict, list[BBoxResult]]:
    """
    Annotated OCR with chunking for documents > 8 pages.

    document_annotation is limited to 8 pages per call. For longer docs we:
      1. Split into 8-page chunks
      2. Run each chunk independently
      3. Merge: collect all bbox results; pick the longest summary as the doc annotation
         (a richer chunk is usually more informative)

    Note: document_annotation and image_annotation come back as JSON strings,
    not Pydantic objects — we parse them explicitly here.
    """
    CHUNK_SIZE = 8
    all_page_indices = list(range(total_pages))
    chunks = [all_page_indices[i:i+CHUNK_SIZE] for i in range(0, total_pages, CHUNK_SIZE)]

    print(f"\n[2/3] Running annotated OCR ({total_pages} pages → {len(chunks)} chunk(s) of ≤{CHUNK_SIZE})...")

    chunk_doc_annotations = []
    all_bbox_results: list[BBoxResult] = []

    for chunk_idx, chunk_pages in enumerate(chunks):
        print(f"    Chunk {chunk_idx+1}/{len(chunks)}: pages {chunk_pages[0]}–{chunk_pages[-1]}")

        response = client.ocr.process(
            model="mistral-ocr-latest",
            pages=chunk_pages,
            document={"type": "document_url", "document_url": doc_url},
            bbox_annotation_format=response_format_from_pydantic_model(BBoxAnnotation),
            document_annotation_format=response_format_from_pydantic_model(DocumentAnnotation),
            include_image_base64=False,
        )

        # document_annotation is a raw JSON string — parse it
        if response.document_annotation:
            try:
                chunk_doc_annotations.append(json.loads(response.document_annotation))
            except json.JSONDecodeError:
                chunk_doc_annotations.append({"raw": response.document_annotation})

        # Collect bbox annotations per page
        for page in response.pages:
            page_markdown = page.markdown or ""
            for img in page.images:
                # image_annotation is also a JSON string — parse it
                annotation = {}
                if img.image_annotation:
                    try:
                        annotation = json.loads(img.image_annotation)
                    except json.JSONDecodeError:
                        annotation = {"raw": img.image_annotation}

                context = extract_bbox_context(page_markdown, img.id)

                all_bbox_results.append(BBoxResult(
                    page_index=page.index,
                    image_id=img.id,
                    coords=(img.top_left_x, img.top_left_y, img.bottom_right_x, img.bottom_right_y),
                    annotation=annotation,
                    context_text=context,
                ))

    # Merge document annotations: prefer the chunk with the longest summary
    # (heuristic — the most content-rich chunk tends to produce the best summary)
    merged_doc = {}
    if chunk_doc_annotations:
        merged_doc = max(chunk_doc_annotations,
                         key=lambda d: len(d.get("summary", "") if isinstance(d, dict) else ""))
        # Collect authors from all chunks (deduplicated)
        all_authors = []
        for d in chunk_doc_annotations:
            if isinstance(d, dict):
                all_authors.extend(d.get("authors", []))
        merged_doc["authors"] = list(dict.fromkeys(all_authors))  # preserve order, dedupe

    print(f"    BBox annotations collected: {len(all_bbox_results)}")
    return merged_doc, all_bbox_results


def print_report(report: OCRReport):
    """Human-readable summary — the kind partners can copy into a Slack update."""

    print("\n" + "="*70)
    print("MISTRAL OCR REPORT")
    print("="*70)

    # Quality gate
    print(f"\n── Quality Gate ({len(report.flagged_pages)} page(s) flagged) ──")
    if not report.flagged_pages:
        print("  All pages passed confidence threshold.")
    for p in report.flagged_pages:
        print(f"  Page {p.page_index}: {p.avg_confidence*100:.1f}% avg confidence")
        if p.low_conf_words:
            print(f"    Low-confidence words: {p.low_conf_words}")

    # Document annotation
    print("\n── Document Annotation ──")
    doc = report.document_annotation
    print(f"  Type     : {doc.get('doc_type', 'n/a')}")
    print(f"  Language : {doc.get('language', 'n/a')}")
    print(f"  Authors  : {', '.join(doc.get('authors', [])) or 'none detected'}")
    print(f"  Summary  :\n    {doc.get('summary', 'n/a')}")

    # BBox annotations with context
    print(f"\n── BBox Annotations ({len(report.bbox_results)} images) ──")
    for b in report.bbox_results:
        ann = b.annotation
        print(f"\n  [{b.image_id}] page {b.page_index} @ ({b.coords[0]:.0f},{b.coords[1]:.0f})→({b.coords[2]:.0f},{b.coords[3]:.0f})")
        print(f"  Type       : {ann.get('image_type', 'n/a')}")
        print(f"  Description: {ann.get('description', 'n/a')[:180]}")
        if b.context_text:
            print(f"  Context    : {b.context_text[:180]}")

    print("\n" + "="*70)


def save_report(report: OCRReport, output_path: Path):
    data = {
        "total_pages": report.total_pages,
        "flagged_pages": [
            {"page": p.page_index, "avg_confidence": round(p.avg_confidence, 4),
             "low_conf_words": p.low_conf_words, "flagged": p.flagged}
            for p in report.flagged_pages
        ],
        "document_annotation": report.document_annotation,
        "bbox_results": [
            {"page": b.page_index, "id": b.image_id,
             "coords": {"x1": b.coords[0], "y1": b.coords[1], "x2": b.coords[2], "y2": b.coords[3]},
             "annotation": b.annotation,
             "context_text": b.context_text}
            for b in report.bbox_results
        ],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nFull report saved → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mistral OCR partner demo script")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", help="Path to local PDF/image file")
    group.add_argument("--url", help="Public URL to a PDF or image")
    parser.add_argument("--confidence-threshold", type=float, default=0.85,
                        help="Pages with avg word confidence below this are flagged (default: 0.85)")
    parser.add_argument("--output", default="ocr_report.json",
                        help="Output JSON file path (default: ocr_report.json)")
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        sys.exit("Error: set MISTRAL_API_KEY environment variable.")

    client = Mistral(api_key=api_key)
    doc_url, label = load_document(args.pdf, args.url)
    print(f"\nDocument: {label}")

    # Step 1 — plain OCR + quality gate
    quality_results, markdown_by_page, pages = run_plain_ocr(client, doc_url, args.confidence_threshold)
    total_pages = len(pages)

    # Step 2 — annotated OCR with chunking
    doc_annotation, bbox_results = run_annotated_ocr_chunked(client, doc_url, total_pages)

    # Step 3 — assemble and print report
    print("\n[3/3] Assembling report...")
    report = OCRReport(
        total_pages=total_pages,
        flagged_pages=[p for p in quality_results if p.flagged],
        document_annotation=doc_annotation,
        bbox_results=bbox_results,
        raw_markdown=markdown_by_page,
    )

    print_report(report)
    save_report(report, Path(args.output))


if __name__ == "__main__":
    main()
