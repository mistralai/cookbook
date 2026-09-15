# Changelog

## Sample documents: OCR 4 showcase cases

Added handwriting, signature, and sparse-table cases to the generated sample documents to
showcase OCR 4, and verified OCR 4 reads each correctly. All handwriting and signatures are
synthetic (font-rendered), so no real person's signature or data is used.

Generated in `inputs/generate_samples.py`; re-run `uv run python inputs/generate_samples.py`
to reproduce.

- **Signature + handwritten date** on the loan application: a cursive signature over the
  signature line and a handwritten date, rendered from script fonts with per-glyph jitter and
  a slight baseline slope.
- **Handwritten note** on the loan application: a short two-line borrower note in the open
  space below the signature.
- **Sparse table** on the pay stub: an "other income and adjustments" grid that is mostly
  empty with a single filled row, the kind of layout that breaks naive text parsers.

New helpers in `generate_samples.py`: `hand()` (jittered handwriting on a rotated transparent
layer) and `sparse_table()` (ruled grid with real empty cells). The signed loan application is
gated by a `signed=True` flag, so the thin/clean/borderline variants stay unsigned for their
own flows.

**Verified:** OCR 4 transcribed the cursive signature, the handwritten date, and both lines
of the handwritten note accurately, and kept the sparse table's column alignment intact
(every column preserved, empty cells empty, the single filled row's values in the right
columns). Verified via `src/ocr_client.ocr_with_annotation`.

## Chat-with-my-document: NVIDIA 10-Q sample

Added a real NVIDIA Form 10-Q as a doc-chat sample (`inputs/nvidia_10q_financials.pdf`) and
wired it into the doc-chat tray (`DOC_SAMPLES` in `app.py`, listed first). Sourced from the
public SEC EDGAR filing (quarter ended 2026-07-26), rendered to PDF and trimmed to the cover
page through the condensed consolidated financial statements (18 pages: income statement,
balance sheet, shareholders' equity, cash flows).

Source (public): https://www.sec.gov/Archives/edgar/data/1045810/000104581026000075/nvda-20260726.htm

**Verified:** doc-chat OCR'd the 18-page filing in a few seconds with the financial tables
preserved, and the document-chat agent answered grounded questions with the correct figures
from the filing (for example, total revenue and total current assets for the quarter, both
matching the filed statements).

## Infrastructure: model capacity defaults

Raised the OCR deployment's default capacity in `infra/main.bicep` from 1 to 10, so a fresh
deploy does not hit 429 throttling when the demo uploads several documents in quick
succession. For large-context doc-chat (for example, chatting over the 10-Q), the chat model
benefits from higher capacity as well; raise `mediumCapacity` if you rely on that path.
