# Mistral OCR-4 Repository Review & Regression Testing Plan

## Executive Summary

This document provides a **comprehensive review** of the OCR-4 capabilities demonstrated in this repository, identifies **corner case gaps**, and proposes a **regression testing framework** to ensure OCR-4 capabilities are fully validated across all document types and edge cases.

---

## 1. Current Coverage Analysis

### 1.1 Document Types Covered ✅

| Document Type | Sample Files | Notebooks | Status |
|---------------|--------------|-----------|--------|
| **PDF** | `mistral7b.pdf`, `Nvidia-10-Q-Form.pdf`, `sparse_income_statement_with_net_income.pdf`, `0000950170-25-100226.pdf` | All | ✅ Covered |
| **PNG Images** | `receipt.png` | `ocr4_comprehensive_showcase.ipynb`, `mistral-docai-ocr-4-0-post-release-checks.ipynb` | ✅ Covered |
| **Word (DOCX)** | `TranscriptFY25q4.docx` | `test_mistral_docai_4_0.py`, `nvidia_10q_ocr4_analysis.ipynb` | ✅ Covered |
| **PowerPoint (PPTX)** | `sample.pptx` | `test_mistral_docai_4_0.py` | ✅ Covered |
| **EPUB** | `minimal.epub` | `test_mistral_docai_4_0.py` | ✅ Covered |

### 1.2 OCR-4 Features Demonstrated ✅

#### Core Capabilities
- [x] **Text Extraction**: Basic OCR from images and documents
- [x] **Bounding Box Detection**: Pixel-precise coordinates via `include_blocks: true`
- [x] **Block Classification**: 13 semantic block types (title, text, table, image, list, equation, etc.)
- [x] **Confidence Scores**: Per-page confidence via `confidence_scores_granularity: 'page'`
- [x] **Table Extraction**: HTML and Markdown table formats
- [x] **Header/Footer Extraction**: via `extract_header: true`, `extract_footer: true`

#### Advanced Capabilities
- [x] **Sparse Table Handling**: Empty cell preservation (`spares_table_ocr.ipynb`)
- [x] **Multilingual Support**: Auto-detection demonstrated
- [x] **Visual Debugging**: Bounding box overlays on source images
- [x] **Confidence Visualization**: Bar charts, heatmaps, summary statistics
- [x] **Financial Document Parsing**: Income statements, 10-Q forms

#### Domain-Specific
- [x] **Financial**: Sparse income statements, 10-Q forms
- [x] **Receipts**: Parking receipts, general receipts
- [x] **Legal**: SEC filings (MS 8-K)
- [x] **Technical**: Whitepapers (mistral7b.pdf)
- [x] **Presentations**: PowerPoint slides

---

## 2. Corner Case Gap Analysis ❌

### 2.1 Document Quality Corner Cases

| Corner Case | Current Coverage | Sample Available | Recommendation |
|-------------|------------------|-----------------|----------------|
| **Low Resolution (<150 DPI)** | ❌ Not tested | No | Add low-res sample |
| **Blurry/Out-of-Focus** | ❌ Not tested | No | Add blurry PDF |
| **Rotated Documents (90°, 180°, 270°)** | ❌ Not tested | No | Add rotated samples |
| **Skewed/Scanned at Angle** | ❌ Not tested | No | Add skewed PDF |
| **Watermarked Documents** | ❌ Not tested | No | Add watermarked sample |
| **Password-Protected PDFs** | ❌ Not tested | No | Test error handling |
| **Corrupted PDFs** | ❌ Not tested | No | Test graceful failure |
| **Very Large Documents (>100 pages)** | ❌ Not tested | No | Add large PDF |

### 2.2 Content Corner Cases

| Corner Case | Current Coverage | Sample Available | Recommendation |
|-------------|------------------|-----------------|----------------|
| **Nested Tables** | ❌ Not tested | No | Add nested table sample |
| **Multi-Column Layouts** | ❌ Not tested | No | Add newspaper-style PDF |
| **Handwritten Text** | ❌ Not tested | No | Add handwritten note |
| **Mathematical Equations** | ⚠️ Mentioned (equation block type) | No | Add LaTeX sample |
| **Code Snippets** | ⚠️ Mentioned (code block type) | No | Add code-heavy PDF |
| **Checkboxes/Radio Buttons** | ❌ Not tested | No | Add form with checkboxes |
| **Barcodes/QR Codes** | ❌ Not tested | No | Add barcode sample |
| **Signatures** | ⚠️ Mentioned (signature block type) | No | Add signed document |
| **Redacted Content** | ❌ Not tested | No | Add redacted PDF |
| **Stamps/Seals** | ❌ Not tested | No | Add stamped document |

### 2.3 Language & Encoding Corner Cases

| Corner Case | Current Coverage | Sample Available | Recommendation |
|-------------|------------------|-----------------|----------------|
| **Mixed Language Documents** | ❌ Not tested | No | Add multilingual invoice |
| **RTL Languages (Arabic, Hebrew)** | ❌ Not tested | No | Add RTL sample |
| **CJK Characters (Chinese, Japanese, Korean)** | ❌ Not tested | No | Add CJK sample |
| **Special Characters (✓, ✗, €, £, ¥)** | ❌ Not tested | No | Add special chars sample |
| **Non-Latin Scripts (Cyrillic, Devanagari)** | ❌ Not tested | No | Add various scripts |

### 2.4 Formatting Corner Cases

| Corner Case | Current Coverage | Sample Available | Recommendation |
|-------------|------------------|-----------------|----------------|
| **Color vs Black & White** | ❌ Not tested | No | Add both versions |
| **Different Font Types** | ❌ Not tested | No | Add font variety sample |
| **Very Small Text (<8pt)** | ❌ Not tested | No | Add small-text PDF |
| **Very Large Text (>24pt)** | ❌ Not tested | No | Add large-text PDF |
| **Text Over Images** | ❌ Not tested | No | Add overlay sample |
| **Low Contrast Text** | ❌ Not tested | No | Add low-contrast PDF |
| **Bold/Italic/Underline** | ❌ Not tested | No | Add formatted text |
| **Subscripts/Superscripts** | ❌ Not tested | No | Add scientific PDF |

### 2.5 Structural Corner Cases

| Corner Case | Current Coverage | Sample Available | Recommendation |
|-------------|------------------|-----------------|----------------|
| **Multi-Level Headers** | ❌ Not tested | No | Add complex headers |
| **Footnotes/Endnotes** | ❌ Not tested | No | Add footnote sample |
| **Table of Contents** | ❌ Not tested | No | Add TOC sample |
| **Appendices** | ❌ Not tested | No | Add appendix sample |
| **Page Numbers in Various Formats** | ❌ Not tested | No | Add page number variations |
| **Hyphenated Words** | ❌ Not tested | No | Add hyphenation sample |

### 2.6 Table-Specific Corner Cases

| Corner Case | Current Coverage | Sample Available | Recommendation |
|-------------|------------------|-----------------|----------------|
| **Merged Cells** | ❌ Not tested | No | Add merged cells table |
| **Colspan/Rowspan** | ❌ Not tested | No | Add colspan/rowspan table |
| **Tables Without Borders** | ❌ Not tested | No | Add borderless table |
| **Tables with Diagonal Lines** | ❌ Not tested | No | Add diagonal table |
| **Multi-Page Tables** | ❌ Not tested | No | Add multi-page table |
| **Tables with Images** | ❌ Not tested | No | Add table with embedded images |
| **Rotated Tables** | ❌ Not tested | No | Add rotated table |

### 2.7 Business Document Corner Cases

| Corner Case | Current Coverage | Sample Available | Recommendation |
|-------------|------------------|-----------------|----------------|
| **Multi-Page Invoices** | ❌ Not tested | No | Add multi-page invoice |
| **Credit Notes** | ❌ Not tested | No | Add credit note sample |
| **Purchase Orders** | ❌ Not tested | No | Add PO sample |
| **Shipping Labels** | ❌ Not tested | No | Add shipping label |
| **Packing Slips** | ❌ Not tested | No | Add packing slip |
| **Bank Statements** | ❌ Not tested | No | Add bank statement |
| **Tax Forms** | ❌ Not tested | No | Add tax form |

---

## 3. OCR-4 Capabilities Not Fully Showcased ⚠️

### 3.1 Underutilized Features

| Feature | Current Usage | Potential |
|---------|---------------|-----------|
| **Token-Level Confidence** | Only page-level used | Could validate per-token accuracy |
| **Region-Level Confidence** | Not demonstrated | Could identify low-confidence regions |
| **All 13 Block Types** | Only some tested | Full block type validation needed |
| **JSON Schema Extraction** | Only bbox_annotation shown | Custom schema testing needed |
| **Image Base64 Inclusion** | Available but not stress-tested | Large image handling validation |

### 3.2 Missing Validation Metrics

- **Precision/Recall for table detection**
- **F1 Score for block classification**
- **IOU (Intersection over Union) for bounding boxes**
- **Character Error Rate (CER)** for text extraction
- **Word Error Rate (WER)** for text extraction
- **Layout Accuracy** (positional fidelity)

---

## 4. Regression Testing Framework

### 4.1 Test Architecture

```
ocr4_regression_tests/
├── __init__.py
├── conftest.py                # Fixtures, test setup
├── test_suites/
│   ├── test_basic_ocr.py        # Text extraction accuracy
│   ├── test_document_types.py   # PDF, PNG, DOCX, PPTX, EPUB
│   ├── test_table_extraction.py # Tables (sparse, nested, complex)
│   ├── test_confidence_scores.py # Token/page/region confidence
│   ├── test_block_classification.py # 13 block types
│   ├── test_bounding_boxes.py  # Precision, recall, IOU
│   ├── test_multilingual.py   # 100+ languages
│   ├── test_edge_cases.py      # All corner cases from Section 2
│   └── test_performance.py     # Speed, throughput, memory
├── test_data/
│   ├── documents/
│   │   ├── financial/
│   │   ├── legal/
│   │   ├── technical/
│   │   └── edge_cases/
│   └── ground_truth/
│       └── *.json           # Expected outputs
├── reports/
│   └── test_report.html      # Generated HTML report
└── requirements.txt
```

### 4.2 Test Data Structure

Each test document should have:
- **Source file** (PDF, PNG, DOCX, etc.)
- **Ground truth JSON** with:
  - Expected text content
  - Expected table structures
  - Expected block types and coordinates
  - Expected confidence thresholds

```json
{
  "document_id": "invoice_001",
  "type": "pdf",
  "language": "en",
  "expected_text": "...",
  "expected_tables": [
    {
      "header": ["Item", "Qty", "Price"],
      "rows": [["Widget", "10", "$100"], ...],
      "empty_cells_preserved": true
    }
  ],
  "expected_blocks": [
    {"type": "title", "text": "INVOICE", "coordinates": {...}},
    {"type": "table", "coordinates": {...}}
  ],
  "confidence_thresholds": {
    "min_page_confidence": 0.85,
    "avg_page_confidence": 0.90
  }
}
```

### 4.3 Test Cases by Category

#### 4.3.1 Basic OCR Tests

| Test ID | Description | Document Type | Metric |
|---------|-------------|---------------|--------|
| OCR-001 | Basic text extraction (English) | PDF | CER < 0.01 |
| OCR-002 | Basic text extraction (French) | PDF | CER < 0.01 |
| OCR-003 | Basic text extraction (German) | PDF | CER < 0.01 |
| OCR-004 | Mixed font sizes | PDF | CER < 0.01 |
| OCR-005 | Bold/italic text | PDF | CER < 0.01 |

#### 4.3.2 Document Type Tests

| Test ID | Description | Document Type | Metric |
|---------|-------------|---------------|--------|
| DOC-001 | Simple PDF | PDF | Success |
| DOC-002 | Image-based PDF | PDF | Success |
| DOC-003 | Scanned PDF | PDF | Success |
| DOC-004 | PNG image | PNG | Success |
| DOC-005 | JPG image | JPG | Success |
| DOC-006 | Word document | DOCX | Success |
| DOC-007 | PowerPoint | PPTX | Success |
| DOC-008 | EPUB | EPUB | Success |
| DOC-009 | Multi-page PDF (10 pages) | PDF | All pages processed |
| DOC-010 | Multi-page PDF (100+ pages) | PDF | All pages processed |

#### 4.3.3 Table Extraction Tests

| Test ID | Description | Test Focus | Metric |
|---------|-------------|-----------|--------|
| TBL-001 | Simple table | Structure | 100% cell match |
| TBL-002 | Sparse table (empty cells) | Empty preservation | 100% empty cells |
| TBL-003 | Table with merged cells | Merged cell handling | 100% accuracy |
| TBL-004 | Multi-column layout | Column detection | Correct column count |
| TBL-005 | Nested tables | Hierarchy | Correct nesting |
| TBL-006 | Multi-page table | Continuity | Seamless continuation |
| TBL-007 | Borderless table | Detection | Correct boundaries |
| TBL-008 | Rotated table | Orientation | Correct alignment |

#### 4.3.4 Block Classification Tests

| Test ID | Description | Block Types | Metric |
|---------|-------------|-------------|--------|
| BLK-001 | All 13 block types | All types | 100% classification |
| BLK-002 | Title detection | title | Precision > 0.95 |
| BLK-003 | Text block detection | text | Precision > 0.95 |
| BLK-004 | Table detection | table | Precision > 0.95 |
| BLK-005 | Image detection | image | Precision > 0.95 |
| BLK-006 | List detection | list | Precision > 0.95 |
| BLK-007 | Equation detection | equation | Precision > 0.95 |
| BLK-008 | Caption detection | caption | Precision > 0.95 |

#### 4.3.5 Confidence Score Tests

| Test ID | Description | Metric |
|---------|-------------|--------|
| CONF-001 | Page-level confidence (high quality) | avg > 0.90 |
| CONF-002 | Page-level confidence (low quality) | avg > 0.75 |
| CONF-003 | Minimum confidence threshold | min > 0.70 |
| CONF-004 | Confidence range | range < 0.30 |
| CONF-005 | Per-page confidence consistency | std_dev < 0.10 |

#### 4.3.6 Edge Case Tests

| Test ID | Description | Category | Metric |
|---------|-------------|----------|--------|
| EDGE-001 | Low resolution (150 DPI) | Quality | Success |
| EDGE-002 | Blurry document | Quality | Success |
| EDGE-003 | Rotated 90° | Orientation | Success |
| EDGE-004 | Rotated 180° | Orientation | Success |
| EDGE-005 | Rotated 270° | Orientation | Success |
| EDGE-006 | Watermarked document | Content | Success |
| EDGE-007 | Small text (<8pt) | Formatting | CER < 0.05 |
| EDGE-008 | Large text (>24pt) | Formatting | CER < 0.01 |
| EDGE-009 | Low contrast | Content | Success |
| EDGE-010 | Color document | Content | Success |
| EDGE-011 | Black & white document | Content | Success |
| EDGE-012 | Mixed languages | Language | Success |
| EDGE-013 | RTL language (Arabic) | Language | Success |
| EDGE-014 | CJK characters | Language | Success |
| EDGE-015 | Special characters | Language | Success |

#### 4.3.7 Multilingual Tests

| Test ID | Description | Language | Metric |
|---------|-------------|----------|--------|
| ML-001 | English | en | CER < 0.01 |
| ML-002 | French | fr | CER < 0.01 |
| ML-003 | German | de | CER < 0.01 |
| ML-004 | Spanish | es | CER < 0.01 |
| ML-005 | Chinese | zh | CER < 0.02 |
| ML-006 | Japanese | ja | CER < 0.02 |
| ML-007 | Korean | ko | CER < 0.02 |
| ML-008 | Arabic | ar | CER < 0.02 |
| ML-009 | Hindi | hi | CER < 0.02 |
| ML-010 | Russian | ru | CER < 0.02 |

#### 4.3.8 Performance Tests

| Test ID | Description | Metric | Target |
|---------|-------------|--------|--------|
| PERF-001 | Single page processing time | Latency | < 2s |
| PERF-002 | 10-page document | Throughput | < 10s |
| PERF-003 | 100-page document | Throughput | < 60s |
| PERF-004 | Concurrent requests (10) | Throughput | All complete |
| PERF-005 | Memory usage | Memory | < 500MB |
| PERF-006 | Large image (10MB) | Success | Success |

#### 4.3.9 Business Document Tests

| Test ID | Description | Document Type | Metric |
|---------|-------------|---------------|--------|
| BIZ-001 | Simple invoice | Invoice | 100% field extraction |
| BIZ-002 | Multi-page invoice | Invoice | 100% field extraction |
| BIZ-003 | Credit note | Credit Note | 100% field extraction |
| BIZ-004 | Purchase order | PO | 100% field extraction |
| BIZ-005 | Bank statement | Bank Statement | 100% field extraction |
| BIZ-006 | Tax form | Tax Form | 100% field extraction |
| BIZ-007 | Shipping label | Shipping Label | 100% field extraction |
| BIZ-008 | Packing slip | Packing Slip | 100% field extraction |
| BIZ-009 | Contract | Contract | 100% field extraction |
| BIZ-010 | Financial statement | Financial | 100% field extraction |

#### 4.3.10 Financial Document Tests

| Test ID | Description | Document Type | Metric |
|---------|-------------|---------------|--------|
| FIN-001 | Income statement | P&L | 100% sparse cell preservation |
| FIN-002 | Balance sheet | Balance Sheet | 100% sparse cell preservation |
| FIN-003 | Cash flow statement | Cash Flow | 100% sparse cell preservation |
| FIN-004 | 10-K filing | SEC Filing | 100% table extraction |
| FIN-005 | 10-Q filing | SEC Filing | 100% table extraction |
| FIN-006 | Annual report | Annual Report | 100% table extraction |

---

## 5. Test Implementation Priorities

### Priority 1: Critical Gaps (Must Implement)
1. **Sparse table edge cases** (merged cells, borderless, multi-page)
2. **Multilingual validation** (CJK, RTL, mixed languages)
3. **Document quality variations** (low res, blurry, rotated)
4. **Confidence score validation** (thresholds, consistency)

### Priority 2: High Value (Should Implement)
1. **Business document validation** (invoices, contracts, forms)
2. **Block classification accuracy** (all 13 types)
3. **Bounding box precision** (IOU metrics)
4. **Performance benchmarks** (latency, throughput)

### Priority 3: Nice to Have (Could Implement)
1. **Handwritten text**
2. **Mathematical equations**
3. **Code snippets**
4. **Barcodes/QR codes**
5. **Signatures**

---

## 6. Automated Regression Test Suite

### 6.1 Test Runner Configuration

```python
# conftest.py
import pytest
import os
from pathlib import Path

# Configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"
GROUND_TRUTH_DIR = TEST_DATA_DIR / "ground_truth"
RESULTS_DIR = Path("test_results")

# Azure Mistral Document AI Configuration
@pytest.fixture
def azure_config():
    from dotenv import load_dotenv
    load_dotenv()
    return {
        "endpoint": os.getenv("AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT"),
        "key": os.getenv("AZURE_MISTRAL_DOCUMENT_AI_KEY"),
        "model": os.getenv("AZURE_AI_DEPLOYMENT_NAME"),
    }

# OCR Client Fixture
@pytest.fixture
def ocr_client(azure_config):
    from ocr4_client import OCRClient  # Custom wrapper
    return OCRClient(
        endpoint=azure_config["endpoint"],
        api_key=azure_config["key"],
        model=azure_config["model"]
    )
```

### 6.2 Sample Test Implementation

```python
# test_table_extraction.py
import pytest
import json
from pathlib import Path

class TestTableExtraction:
    @pytest.mark.parametrize("test_file,expected", [
        ("sparse_income_statement_with_net_income.pdf", "ground_truth/sparse_income.json"),
        ("P&L/P&L_1.pdf", "ground_truth/pl_1.json"),
        ("Invoice/invoice_1.pdf", "ground_truth/invoice_1.json"),
    ])
    def test_sparse_table_preservation(self, ocr_client, test_file, expected):
        """Test that empty cells in sparse tables are preserved."""
        # Load test document
        doc_path = Path("test_data/documents") / test_file
        
        # Run OCR
        result = ocr_client.process_file(doc_path)
        
        # Load ground truth
        gt_path = Path("test_data/ground_truth") / expected
        with open(gt_path) as f:
            ground_truth = json.load(f)
        
        # Extract tables
        tables = result.get("tables", [])
        
        # Validate sparse cell preservation
        for gt_table, actual_table in zip(ground_truth["tables"], tables):
            # Check that empty cells are preserved as None/empty
            for row_idx, (gt_row, actual_row) in enumerate(zip(gt_table["rows"], actual_table["rows"])):
                for col_idx, (gt_cell, actual_cell) in enumerate(zip(gt_row, actual_row)):
                    if gt_cell is None:
                        # Empty cell in ground truth should remain empty
                        assert actual_cell in [None, "", " "], \
                            f"Empty cell at row {row_idx}, col {col_idx} was not preserved"
                    else:
                        # Non-empty cell should match
                        assert actual_cell == gt_cell, \
                            f"Cell mismatch at row {row_idx}, col {col_idx}"
        
        assert True  # All checks passed

    def test_table_structure_integrity(self, ocr_client):
        """Test that table structure (rows, columns) is preserved."""
        doc_path = Path("test_data/documents/financial/tables.pdf")
        result = ocr_client.process_file(doc_path, table_format="markdown")
        
        # Parse markdown tables
        for page in result.get("pages", []):
            for table in page.get("tables", []):
                content = table.get("content", "")
                rows = content.strip().split("\n")
                
                # Skip separator rows
                rows = [r for r in rows if not all(c in "|-\s:" for c in r)]
                
                if rows:
                    # All rows should have the same number of columns
                    col_counts = [r.count("|") - 1 for r in rows]
                    assert len(set(col_counts)) == 1, \
                        f"Inconsistent column count: {col_counts}"
```

### 6.3 Performance Test Implementation

```python
# test_performance.py
import pytest
import time
from pathlib import Path

class TestPerformance:
    @pytest.mark.performance
    def test_single_page_latency(self, ocr_client):
        """Test processing time for a single page document."""
        doc_path = Path("test_data/documents/receipt.png")
        
        start_time = time.time()
        result = ocr_client.process_file(doc_path)
        elapsed = time.time() - start_time
        
        assert elapsed < 2.0, f"Single page took {elapsed:.2f}s (target: <2s)"
        assert len(result.get("pages", [])) > 0, "No pages returned"
    
    @pytest.mark.performance
    def test_multi_page_throughput(self, ocr_client):
        """Test processing time for a 10-page document."""
        doc_path = Path("test_data/documents/Nvidia-10-Q-Form-p1-30.pdf")
        
        start_time = time.time()
        result = ocr_client.process_file(doc_path)
        elapsed = time.time() - start_time
        
        page_count = len(result.get("pages", []))
        assert elapsed < 10.0, f"10-page doc took {elapsed:.2f}s (target: <10s)"
        assert page_count >= 10, f"Expected >=10 pages, got {page_count}"
    
    @pytest.mark.performance
    def test_concurrent_requests(self, ocr_client):
        """Test handling of concurrent requests."""
        doc_path = Path("test_data/documents/receipt.png")
        
        start_time = time.time()
        results = [
            ocr_client.process_file(doc_path) for _ in range(10)
        ]
        elapsed = time.time() - start_time
        
        assert all(len(r.get("pages", [])) > 0 for r in results), \
            "Some concurrent requests failed"
        assert elapsed < 15.0, f"10 concurrent requests took {elapsed:.2f}s (target: <15s)"
```

### 6.4 Confidence Score Validation

```python
# test_confidence_scores.py
import pytest

class TestConfidenceScores:
    def test_page_confidence_thresholds(self, ocr_client):
        """Test that confidence scores meet minimum thresholds."""
        doc_path = Path("test_data/documents/high_quality.pdf")
        result = ocr_client.process_file(
            doc_path, 
            confidence_scores_granularity="page"
        )
        
        for page in result.get("pages", []):
            scores = page.get("confidence_scores", {})
            avg = scores.get("average_page_confidence_score")
            min_score = scores.get("minimum_page_confidence_score")
            
            assert avg is not None, f"Missing avg confidence for page {page.get('index')}"
            assert min_score is not None, f"Missing min confidence for page {page.get('index')}"
            
            # High quality documents should have avg confidence > 0.85
            assert avg > 0.85, f"Avg confidence {avg} < 0.85 threshold"
            # Min confidence should be > 0.70
            assert min_score > 0.70, f"Min confidence {min_score} < 0.70 threshold"
    
    def test_confidence_consistency(self, ocr_client):
        """Test that confidence scores are consistent across pages."""
        doc_path = Path("test_data/documents/uniform_quality.pdf")
        result = ocr_client.process_file(
            doc_path,
            confidence_scores_granularity="page"
        )
        
        avg_confidences = [
            p.get("confidence_scores", {}).get("average_page_confidence_score")
            for p in result.get("pages", [])
            if p.get("confidence_scores")
        ]
        
        if len(avg_confidences) > 1:
            # Calculate standard deviation
            import statistics
            std_dev = statistics.stdev(avg_confidences)
            
            # Uniform documents should have std dev < 0.10
            assert std_dev < 0.10, \
                f"High confidence variance: {std_dev:.4f} (target: <0.10)"
```

---

## 7. Test Reporting & CI/CD Integration

### 7.1 Test Report Generation

```python
# Generate HTML report
pytest --html=reports/test_report.html --self-contained-html
```

### 7.2 CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/ocr4_regression.yml
name: OCR-4 Regression Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # Run daily at 2 AM

jobs:
  regression-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest pytest-html pytest-cov requests python-dotenv pandas matplotlib
      
      - name: Run regression tests
        env:
          AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT: ${{ secrets.AZURE_ENDPOINT }}
          AZURE_MISTRAL_DOCUMENT_AI_KEY: ${{ secrets.AZURE_API_KEY }}
          AZURE_AI_DEPLOYMENT_NAME: ${{ secrets.AZURE_MODEL }}
        run: |
          pytest test_suites/ -v --html=reports/test_report.html --self-contained-html
      
      - name: Upload test report
        uses: actions/upload-artifact@v3
        with:
          name: test-report
          path: reports/test_report.html
      
      - name: Check for regressions
        run: |
          # Compare current results with baseline
          # Fail if any key metrics degraded
          python scripts/check_regression.py
```

### 7.3 Baseline Management

```python
# scripts/baseline_manager.py
import json
from pathlib import Path

BASELINE_FILE = Path("test_suites/baselines.json")

class BaselineManager:
    def __init__(self):
        self.baselines = self.load_baselines()
    
    def load_baselines(self):
        if BASELINE_FILE.exists():
            with open(BASELINE_FILE) as f:
                return json.load(f)
        return {}
    
    def update_baseline(self, test_id, metric, value):
        if test_id not in self.baselines:
            self.baselines[test_id] = {}
        self.baselines[test_id][metric] = value
        self.save()
    
    def check_regression(self, test_id, metric, current_value, threshold=0.05):
        """Check if current value represents a regression from baseline."""
        baseline = self.baselines.get(test_id, {}).get(metric)
        if baseline is None:
            return False, "No baseline"
        
        # For metrics where higher is better (accuracy, confidence)
        if metric in ["accuracy", "confidence", "score"]:
            degradation = baseline - current_value
            if degradation > threshold:
                return True, f"Regression: {metric} dropped from {baseline} to {current_value}"
        
        # For metrics where lower is better (CER, WER, latency)
        elif metric in ["cer", "wer", "latency"]:
            degradation = current_value - baseline
            if degradation > threshold:
                return True, f"Regression: {metric} increased from {baseline} to {current_value}"
        
        return False, "OK"
    
    def save(self):
        with open(BASELINE_FILE, 'w') as f:
            json.dump(self.baselines, f, indent=2)
```

---

## 8. Recommendations

### 8.1 Immediate Actions (Next 2 Weeks)

1. **Create test data repository** with all corner case samples
2. **Implement Priority 1 tests** (sparse tables, multilingual, confidence scores)
3. **Set up CI/CD pipeline** for automated regression testing
4. **Generate baseline metrics** for all existing samples

### 8.2 Short-Term Actions (Next Month)

1. **Expand test coverage** to Priority 2 areas
2. **Integrate with existing notebooks** (add test cells to showcases)
3. **Document test results** in a public dashboard
4. **Create performance benchmarks** for different document types

### 8.3 Long-Term Actions (Next Quarter)

1. **Full test automation** with nightly runs
2. **Integration with Azure Test Plans**
3. **Customer-facing test reports** (transparency)
4. **Competitive benchmarking** (vs. AWS Textract, Google Document AI)

---

## 9. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Test Coverage** | >90% | % of features tested |
| **Test Pass Rate** | >95% | % of tests passing |
| **Regression Detection** | <24h | Time to detect regressions |
| **Test Execution Time** | <10 min | Full suite runtime |
| **Baseline Stability** | >99% | % of metrics within threshold |

---

## 10. Conclusion

The current OCR-4 repository demonstrates **strong core capabilities** but has **significant gaps in corner case coverage**. Implementing the proposed **regression testing framework** will:

1. ✅ **Ensure OCR-4 capabilities are fully validated** across all document types and edge cases
2. ✅ **Prevent regressions** as the model evolves
3. ✅ **Provide competitive differentiation** with published benchmark results
4. ✅ **Build customer trust** through transparent testing
5. ✅ **Accelerate adoption** with proven reliability

**Next Step:** Implement Priority 1 tests and establish baseline metrics within 2 weeks.
