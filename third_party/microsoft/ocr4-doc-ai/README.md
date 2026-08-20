# Azure Mistral OCR-4 Cookbook

Examples and validation notebooks for Mistral OCR-4 hosted through Azure AI Foundry. The notebooks submit local images and documents to an Azure OCR endpoint, inspect the returned Markdown and structured data, and demonstrate layout analysis, annotations, confidence scores, multilingual OCR, and sparse-table handling.

## What This Directory Contains

| File | Purpose |
| --- | --- |
| `ocr4_utils.py` | Shared configuration, encoding, OCR request, annotation, parsing, confidence, and plotting helpers. |
| `ocr4_comprehensive_showcase.ipynb` | Broad feature tour covering bounding boxes, block classification, confidence scores, and multilingual input. |
| `handwriting_signature.ipynb` | OCR and structured extraction for `check.png` and `handwritten_sample.png`, including signature extraction and annotated plots. |
| `language_support.ipynb` | Persian handwriting OCR using `persian-writing.jpeg` and `persian_script.jpeg`. |
| `mistral-docai-ocr-4-0-post-release-checks.ipynb` | Basic post-release checks for image, PDF, DOCX, PPTX, and EPUB inputs. |
| `nvidia_10q_ocr4_analysis.ipynb` | Markdown, table, paragraph bounding-box, classification, and confidence analysis for an Nvidia 10-Q PDF. |
| `spares_table_ocr.ipynb` | Sparse-table parsing with missing-cell preservation and PDF text fallback. |
| `test_mistral_docai_4_0.py` | Pytest coverage for encoding, annotation payloads, response parsing, and live document requests. |
| `samples/` | Local images and documents used by the examples. |
| `env.example` | Template for the required Azure configuration variables. |

## Features Demonstrated

### OCR and Markdown

- Base64 encoding of local files.
- Data URL construction with MIME-type detection.
- OCR-4 Markdown extraction from images, PDFs, DOCX, PPTX, and EPUB files.
- Optional header, footer, table, image, and block extraction.

### Bounding Boxes and Classification

When `include_blocks` is enabled, OCR-4 returns pixel coordinates for recognized regions:

- `top_left_x`, `top_left_y`
- `bottom_right_x`, `bottom_right_y`
- semantic `type`
- extracted `content`

The shared plotting helpers support both raster images and PDF pages:

- `plot_bounding_boxes()` overlays blocks on an image file.
- `plot_page_bounding_boxes()` draws blocks using OCR page dimensions and is safe for PDF responses.
- `BLOCK_COLORS` provides consistent colors for semantic types such as `title`, `text`, `table`, `image`, `signature`, `header`, and `footer`.

### Confidence Scores

Set `confidence_scores_granularity` to one of:

- `page`: average and minimum confidence for each page.
- `word`: page-level statistics plus per-word scores where supported.

Confidence values range from `0` to `1`. The examples display average and minimum scores and use them to identify potentially difficult regions.

### Structured Annotations

The notebooks demonstrate `document_annotation_format` and `bbox_annotation_format` using JSON schemas. Annotation results may be returned as dictionaries or JSON strings; `parse_annotation()` normalizes either form.

The check example explicitly inspects the handwritten authorized-signature line and retrieves the visible signer name, such as `Evelyn Buchanan`.

### Sparse Tables

`markdown_table_to_df()` pads short rows and preserves blank cells as missing values. `extract_net_income_values_from_pdf()` provides a source-PDF fallback for sparse rows when OCR table alignment is ambiguous.

## Prerequisites

- Python 3.10 or newer. Python 3.14 is used by the current workspace.
- An Azure AI Foundry deployment of Mistral OCR-4.
- Azure endpoint, API key, and deployment name.
- Jupyter support in VS Code or another Jupyter client.
- Network access to the configured Azure endpoint for OCR requests.

## Installation

From this directory, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the packages used by the notebooks and tests:

```bash
python -m pip install --upgrade pip
python -m pip install \
  ipykernel \
  jupyter \
  matplotlib \
  pandas \
  pillow \
  pymupdf \
  pytest \
  python-dotenv \
  requests
```

Register the environment as a Jupyter kernel if needed:

```bash
python -m ipykernel install --user --name ocr4-doc-ai --display-name "Python (OCR-4)"
```

Select `Python (OCR-4)` as the kernel in VS Code before running a notebook.

## Configuration

Copy the example file to `.env` and replace the placeholder values:

```bash
cp env.example .env
```

Required variables:

```dotenv
AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT=https://<your-foundry-instance>.services.ai.azure.com/...
AZURE_MISTRAL_DOCUMENT_AI_KEY=<your-api-key>
AZURE_AI_DEPLOYMENT_NAME=mistral-ocr-4-0
```

`OCRConfig.from_env()` loads and validates these values. Do not commit `.env`, API keys, or other credential files. The notebooks use `python-dotenv` to load environment variables locally.

## Running the Notebooks

Open a notebook in VS Code, select the configured Jupyter kernel, and run cells from top to bottom. This order matters because later cells consume responses and variables produced earlier.

Recommended starting points:

1. `ocr4_comprehensive_showcase.ipynb` for a complete feature overview.
2. `handwriting_signature.ipynb` for image OCR, signatures, boxes, classifications, and confidence visualization.
3. `spares_table_ocr.ipynb` for missing-cell-safe table processing.
4. `nvidia_10q_ocr4_analysis.ipynb` for large-document layout and confidence analysis.
5. `language_support.ipynb` for Persian handwriting.
6. `mistral-docai-ocr-4-0-post-release-checks.ipynb` for format coverage.

The notebooks use relative paths such as `samples/check.png`. Run them with the notebook working directory set to this directory. If paths resolve incorrectly, change the notebook working directory or open the directory as the VS Code workspace.

## Shared Module API

The main reusable functions in `ocr4_utils.py` are:

```python
from ocr4_utils import (
    OCRConfig,
    build_ocr_payload,
    call_ocr4,
    encode_file,
    extract_text,
    ocr_request,
    page_confidence,
    plot_bounding_boxes,
    plot_page_bounding_boxes,
)
```

Typical image OCR usage:

```python
from pathlib import Path
from ocr4_utils import OCRConfig, call_ocr4, page_confidence, plot_bounding_boxes

config = OCRConfig.from_env()
image_path = Path('samples/check.png')
response = call_ocr4(image_path, config)
page = response['pages'][0]

plot_bounding_boxes(
    image_path,
    page.get('blocks', []),
    title='Check OCR-4 layout',
    confidence=page_confidence(page),
)
```

Typical PDF layout usage:

```python
from ocr4_utils import OCRConfig, build_ocr_payload, ocr_request, page_confidence, plot_page_bounding_boxes

config = OCRConfig.from_env()
payload = build_ocr_payload(
    'samples/mistral7b.pdf',
    config,
    include_blocks=True,
    confidence_granularity='page',
)
response = ocr_request(payload, config)
plot_page_bounding_boxes(
    response['pages'][0],
    title='PDF OCR-4 layout',
    confidence=page_confidence(response['pages'][0]),
)
```

For a PDF, use `plot_page_bounding_boxes()` rather than `plot_bounding_boxes()`. The latter calls PIL image loading and cannot open a PDF directly.

## Testing

Compile the shared module:

```bash
python3 -m py_compile ocr4_utils.py
```

Validate notebook JSON:

```bash
python3 -c "import json; from pathlib import Path; [json.loads(p.read_text(encoding='utf-8')) for p in Path('.').glob('*.ipynb')]; print('all notebooks valid JSON')"
```

Run the test suite:

```bash
python3 -m pytest -q
```

The test file contains both local unit-style tests and live Azure integration tests. Live tests require valid environment variables, network access, and the sample documents. To run only tests that do not call Azure, select the encoding, payload, and mocked response tests explicitly or add markers as the suite evolves.

## Troubleshooting

### Missing environment variables

`OCRConfig.from_env()` raises a `RuntimeError` listing missing variables. Confirm that `.env` exists in the working directory or export the variables in the shell before starting the notebook kernel.

### Stale imports in a notebook kernel

If a newly added function raises `ImportError`, the kernel may have cached an older version of `ocr4_utils`. Restart the kernel, or reload the module before importing:

```python
import importlib
import ocr4_utils

ocr4_utils = importlib.reload(ocr4_utils)
```

### `UnidentifiedImageError` for a PDF

A PDF is not a raster image. Do not pass a `.pdf` path to `PIL.Image.open()` or to `plot_bounding_boxes()`. Use `plot_page_bounding_boxes()` with the OCR response page instead. For a visual PDF preview, render the page with PyMuPDF first.

### `NameError` for a derived variable

Notebook cells are stateful. Run cells from the beginning, especially the cell that creates values such as `response`, `pages`, `pdf_lines`, or `net_income_row`. Restarting the kernel and running all cells is the most reliable way to reproduce results.

### Package import failures

Install dependencies into the interpreter selected by the notebook, not only into another system Python:

```bash
python -m pip install matplotlib pandas pillow pymupdf python-dotenv requests
```

If matplotlib reports a missing `packaging` dependency, install it explicitly:

```bash
python -m pip install packaging
```

## Outputs and Generated Files

Some notebooks display results inline. The Nvidia analysis notebook can also generate:

- `nvidia_10q_ocr4_output.md`

Generated outputs may be large or environment-specific. Review them before committing or sharing.

## Security and Data Handling

- Treat the Azure API key as a secret.
- Do not place secrets in notebooks, source files, output cells, screenshots, or commits.
- Local files are base64-encoded and sent to the configured Azure endpoint.
- Confirm that sample documents are permitted to leave the local machine before running OCR.
- Avoid printing full API responses when they contain confidential document content.

## Project Notes

The notebooks intentionally keep document-specific analysis close to the example that uses it, while common transport, encoding, schema, parsing, and visualization behavior lives in `ocr4_utils.py`. This keeps examples readable and makes fixes to shared OCR behavior available across the directory.
