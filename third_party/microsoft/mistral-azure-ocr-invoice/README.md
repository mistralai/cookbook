# 🏥 Invoice OCR Analyzer

**Mistral Document AI-powered OCR extraction, correction, and invoice analysis of PDF documents.**

## Features

| Feature | Description |
|---------|-------------|
| **OCR Extraction** | Mistral Document AI (Azure-hosted) extracts text, tables, and images from PDF documents |
| **OCR Error Correction** | Automatic fixing of common OCR errors in Invoice units|
| **Completeness Checking** | Detects skipped rows, empty cells, and sequential gaps in extracted tables |
| **Invoice Value Analysis** | Extracts structured lab values and classifies them against 60+ standard clinical reference ranges |
| **Anomaly Highlighting** | Color-coded status badges: Normal (green), Low/High (orange), Critical (red) |
| **Interactive Web UI** | Gradio-based interface with PDF viewer, results panel, corrections log, and reference table |

## Architecture

```
PDF Upload
    │
    ▼
┌────────────────────┐
│  Mistral OCR       │  ← Azure endpoint (REST + SDK fallback)
│  (ocr_engine.py)   │
└────────┬───────────┘
         │ raw markdown / HTML tables
         ▼
┌────────────────────┐
│  Error Correction  │  ← regex rules for Invoice OCR errors
│  (ocr_correction)  │
└────────┬───────────┘
         │ corrected text
         ▼
┌────────────────────┐
│  Completeness      │  ← Skip detection, empty-row checks
│  Check             │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Invoice Analysis  │  ← Extract lab values, compare to reference ranges
│  (invoice_analyzer)│
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Gradio Web UI     │  ← Results, PDF viewer, corrections log
│  (app.py)          │
└────────────────────┘
```


## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web application
python app.py
```

Then open **http://localhost:7860** in your browser.

## Configuration

Set via environment variables or edit `config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MISTRAL_OCR_ENDPOINT` | Azure endpoint | Full OCR endpoint URL |
| `MISTRAL_API_KEY` | (set in config) | Azure AI / Mistral API key |
| `MISTRAL_OCR_MODEL` | `ocr` | Model deployment name |
| `MAX_PAGES` | `0` (all) | Limit pages to process |

## OCR Error Corrections

The system automatically fixes these common invoice OCR errors:

| Error Pattern | Correction | Example |
|--------------|------------|---------|
| `l` ↔ `1` in units | Unit-aware fix | `mg/1` → `mg/l`, `g/d1` → `g/dl` |
| `O` ↔ `0` in numbers | Context-aware fix | `O.5` → `0.5` |
| Corrupted terms | Dictionary fix | `ce11s` → `cells`, `p1ate1ets` → `platelets` |
| `mmH9` → `mmHg` | Unit fix | Blood pressure units |
| Whitespace issues | Normalization | `3 .5` → `3.5` |


## Project Structure

```
├── app.py                 # Gradio web UI (entry point)
├── config.py              # Configuration (endpoint, keys)
├── ocr_engine.py          # Mistral Document AI OCR extraction
├── ocr_correction.py      # Error correction + completeness checking
├── invoice_analyzer.py    # Reference ranges + value analysis
├── requirements.txt       # Python dependencies
├── data/                  # Sample PDFs
└── uploads/               # Temporary upload storage
```
