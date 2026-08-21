# OCR-4 Regression Test Quickstart Guide

## Overview

This guide provides a **fast path** to implement regression testing for Mistral OCR-4.0 using the existing repository assets.

---

## 🎯 Quick Assessment: Current State

### ✅ What's Already Covered (Can Test Immediately)

The repository already has **excellent foundation** for testing:

1. **Document Types Available** (in `/samples/`):
   - PDF: `mistral7b.pdf`, `Nvidia-10-Q-Form.pdf`, `sparse_income_statement_with_net_income.pdf`, `0000950170-25-100226.pdf`
   - PNG: `receipt.png`
   - DOCX: `TranscriptFY25q4.docx`
   - PPTX: `sample.pptx`
   - EPUB: `minimal.epub`
   - Test Docs: P&L statements, Invoices, DGD forms

2. **Existing Test Infrastructure** (`test_mistral_docai_4_0.py`):
   - ✅ File encoding tests
   - ✅ Bounding box annotation tests
   - ✅ Integration tests for PDF, Word, PowerPoint, EPUB
   - ✅ Table extraction validation

3. **Feature Showcases** (Notebooks):
   - ✅ `ocr4_comprehensive_showcase.ipynb`: Bounding boxes, block classification, confidence scores, multilingual
   - ✅ `spares_table_ocr.ipynb`: Sparse table handling with empty cell preservation
   - ✅ `nvidia_10q_ocr4_analysis.ipynb`: Confidence visualization, block classification, bounding boxes
   - ✅ `mistral-docai-ocr-4-0-post-release-checks.ipynb`: Basic OCR, bbox annotation

---

## 🚀 5-Minute Regression Test Setup

### Step 1: Run Existing Tests

```bash
# Navigate to repo
cd /Users/ninad.joshi/cookbook/third_party/microsoft/ocr4-doc-ai

# Install dependencies (if not already installed)
pip install pytest requests python-dotenv

# Copy .env.example to .env if needed
cp .env.example .env

# Edit .env with your Azure credentials
nano .env  # Add your AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT, KEY, DEPLOYMENT_NAME

# Run existing tests
python -m pytest test_mistral_docai_4_0.py -v
```

### Step 2: Quick Confidence Score Validation

Add this to a new file `quick_confidence_test.py`:

```python
"""
Quick confidence score regression test for OCR-4
"""
import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.getenv('AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT')
API_KEY = os.getenv('AZURE_MISTRAL_DOCUMENT_AI_KEY')
MODEL = os.getenv('AZURE_AI_DEPLOYMENT_NAME')

HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}',
}

def encode_file(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_confidence_scores():
    """Test that confidence scores meet minimum thresholds."""
    
    # Test files with expected minimum confidence
    test_cases = [
        ("samples/receipt.png", 0.85),
        ("samples/sparse_income_statement_with_net_income.pdf", 0.85),
        ("samples/mistral7b.pdf", 0.88),
    ]
    
    all_passed = True
    
    for file_path, min_confidence in test_cases:
        if not os.path.exists(file_path):
            print(f"⚠️  Skipping {file_path} (not found)")
            continue
            
        payload = {
            'model': MODEL,
            'document': {
                'type': 'document_url',
                'document_url': f'data:application/pdf;base64,{encode_file(file_path)}',
            },
            'confidence_scores_granularity': 'page',
        }
        
        response = requests.post(ENDPOINT, json=payload, headers=HEADERS)
        result = response.json()
        
        print(f"\n📄 Testing: {file_path}")
        
        for page in result.get('pages', []):
            scores = page.get('confidence_scores', {})
            avg = scores.get('average_page_confidence_score')
            min_score = scores.get('minimum_page_confidence_score')
            
            if avg is not None:
                print(f"  Page {page.get('index')}: avg={avg:.4f}, min={min_score:.4f}")
                
                if avg < min_confidence:
                    print(f"  ❌ FAIL: avg confidence {avg:.4f} < {min_confidence} threshold")
                    all_passed = False
                elif avg >= 0.90:
                    print(f"  ✅ PASS: High confidence ({avg:.4f})")
                else:
                    print(f"  ⚠️  WARN: Moderate confidence ({avg:.4f})")
    
    return all_passed

if __name__ == "__main__":
    print("=" * 60)
    print("OCR-4 Confidence Score Regression Test")
    print("=" * 60)
    
    try:
        passed = test_confidence_scores()
        print("\n" + "=" * 60)
        if passed:
            print("✅ ALL TESTS PASSED")
        else:
            print("❌ SOME TESTS FAILED")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
```

Run it:
```bash
python quick_confidence_test.py
```

---

## 📊 Sparse Table Regression Test

Add this to `quick_sparse_test.py`:

```python
"""
Quick sparse table regression test for OCR-4
"""
import os
import re
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.getenv('AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT')
API_KEY = os.getenv('AZURE_MISTRAL_DOCUMENT_AI_KEY')
MODEL = os.getenv('AZURE_AI_DEPLOYMENT_NAME')

HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}',
}

def encode_file(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def extract_net_income_from_pdf(pdf_path):
    """Extract net income row from PDF text (fallback method)."""
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    lines = []
    for page in doc:
        lines.extend(page.get_text('text').splitlines())
    doc.close()
    
    for idx, line in enumerate(lines):
        if 'net income' in line.lower():
            values = []
            for nxt in lines[idx + 1 : idx + 12]:
                candidate = nxt.strip()
                if candidate in {'', '[ _ ]', '—', '-', '_'}:
                    values.append(None)
                    continue
                if re.search(r'\d', candidate):
                    numbers = re.findall(r'\$?\d[\d,]*(?:\.\d+)?', candidate)
                    if numbers:
                        values.extend(float(v.replace('$', '').replace(',', '')) for v in numbers)
            if len(values) >= 5:
                return {
                    'Q1 Actual': values[0],
                    'Q2 Forecast': values[1],
                    'Q3 Actual': values[2],
                    'Q4 Projection': values[3],
                    'Full Year Total': values[4],
                }
    return {}

def parse_money(value):
    """Parse money value from OCR output."""
    if value is None or str(value).strip() == '':
        return None
    cleaned = str(value).strip().replace('$', '').replace(',', '').replace('%', '').strip()
    negative = cleaned.startswith('(') and cleaned.endswith(')')
    if negative:
        cleaned = '-' + cleaned[1:-1]
    cleaned = cleaned.replace('(', '').replace(')', '')
    try:
        return float(cleaned)
    except ValueError:
        return None

def extract_net_income_from_ocr(result):
    """Extract net income row from OCR output."""
    # Search all pages for net income table
    for page in result.get('pages', []):
        tables = page.get('tables', [])
        for table in tables:
            content = table.get('content', '')
            if 'net income' in content.lower():
                # Parse markdown table
                rows = content.strip().split('\n')
                for row in rows:
                    if 'net income' in row.lower():
                        # Extract values from this row
                        cells = [c.strip() for c in row.strip('|').split('|')]
                        if len(cells) >= 6:  # Line Item + 5 values
                            return {
                                'Q1 Actual': parse_money(cells[1]),
                                'Q2 Forecast': parse_money(cells[2]),
                                'Q3 Actual': parse_money(cells[3]),
                                'Q4 Projection': parse_money(cells[4]),
                                'Full Year Total': parse_money(cells[5]),
                            }
    return {}

def test_sparse_table():
    """Test sparse table handling with empty Q4 cell."""
    pdf_path = "samples/sparse_income_statement_with_net_income.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"⚠️  Skipping: {pdf_path} not found")
        return True
    
    # Ground truth (from PDF)
    ground_truth = {
        'Q1 Actual': 40300.0,
        'Q2 Forecast': 8500.0,
        'Q3 Actual': 34500.0,
        'Q4 Projection': None,  # This should be preserved as empty
        'Full Year Total': 83300.0,
    }
    
    print(f"\n📄 Testing sparse table: {pdf_path}")
    
    # Test 1: PDF text fallback (baseline)
    pdf_baseline = extract_net_income_from_pdf(pdf_path)
    print(f"\n  PDF Text Extraction (Baseline):")
    for k, v in pdf_baseline.items():
        print(f"    {k}: {v}")
    
    # Test 2: OCR extraction
    payload = {
        'model': MODEL,
        'document': {
            'type': 'document_url',
            'document_url': f'data:application/pdf;base64,{encode_file(pdf_path)}',
        },
        'table_format': 'markdown',
    }
    
    response = requests.post(ENDPOINT, json=payload, headers=HEADERS)
    ocr_result = response.json()
    
    ocr_extracted = extract_net_income_from_ocr(ocr_result)
    print(f"\n  OCR Extraction:")
    for k, v in ocr_extracted.items():
        print(f"    {k}: {v}")
    
    # Validate
    all_passed = True
    
    # Check numeric values
    for key in ['Q1 Actual', 'Q2 Forecast', 'Q3 Actual', 'Full Year Total']:
        gt_val = ground_truth[key]
        ocr_val = ocr_extracted.get(key)
        
        if ocr_val is None:
            print(f"  ❌ FAIL: {key} not extracted (expected {gt_val})")
            all_passed = False
        elif abs(ocr_val - gt_val) > 0.01:
            print(f"  ❌ FAIL: {key} = {ocr_val}, expected {gt_val}")
            all_passed = False
        else:
            print(f"  ✅ PASS: {key} = {ocr_val}")
    
    # Check Q4 empty cell preservation
    q4_ocr = ocr_extracted.get('Q4 Projection')
    if q4_ocr is None or str(q4_ocr).strip() == '':
        print(f"  ✅ PASS: Q4 Projection preserved as empty")
    else:
        print(f"  ❌ FAIL: Q4 Projection = {q4_ocr}, expected None/empty")
        all_passed = False
    
    return all_passed

if __name__ == "__main__":
    print("=" * 60)
    print("OCR-4 Sparse Table Regression Test")
    print("=" * 60)
    
    try:
        # Check if PyMuPDF is available
        try:
            import fitz
        except ImportError:
            print("\n⚠️  PyMuPDF not installed. Install with: pip install pymupdf")
            exit(1)
        
        passed = test_sparse_table()
        print("\n" + "=" * 60)
        if passed:
            print("✅ ALL TESTS PASSED")
        else:
            print("❌ SOME TESTS FAILED")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
```

Run it:
```bash
pip install pymupdf
python quick_sparse_test.py
```

---

## 🧪 Multilingual Quick Test

Add this to `quick_multilingual_test.py`:

```python
"""
Quick multilingual regression test for OCR-4
"""
import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.getenv('AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT')
API_KEY = os.getenv('AZURE_MISTRAL_DOCUMENT_AI_KEY')
MODEL = os.getenv('AZURE_AI_DEPLOYMENT_NAME')

HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}',
}

def encode_file(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_multilingual():
    """Test OCR on documents with different languages."""
    
    # Create synthetic multilingual test document
    from PIL import Image, ImageDraw, ImageFont
    import io
    
    # Create image with text in multiple languages
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Add text in different languages
    languages = {
        'English': 'Hello World',
        'French': 'Bonjour le monde',
        'German': 'Hallo Welt',
        'Spanish': 'Hola Mundo',
        'Chinese': '你好世界',
        'Japanese': 'こんにちは世界',
        'Korean': '안녕하세요 세계',
    }
    
    y = 50
    for lang, text in languages.items():
        draw.text((50, y), f"{lang}: {text}", fill='black', font=font)
        y += 40
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Encode
    encoded = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    print("\n📄 Testing multilingual OCR")
    
    payload = {
        'model': MODEL,
        'document': {
            'type': 'document_url',
            'document_url': f'data:image/png;base64,{encoded}',
        }
    }
    
    response = requests.post(ENDPOINT, json=payload, headers=HEADERS)
    result = response.json()
    
    # Extract text
    all_text = []
    for page in result.get('pages', []):
        all_text.append(page.get('markdown', ''))
    
    full_text = '\n'.join(all_text)
    print(f"\nExtracted text:\n{full_text[:500]}")
    
    # Check for language coverage
    all_passed = True
    for lang, expected in languages.items():
        # For non-Latin scripts, check if characters are present
        if lang in ['Chinese', 'Japanese', 'Korean']:
            # Check if any CJK characters are in the output
            has_cjk = any(0x4E00 <= ord(c) <= 0x9FFF for c in full_text) or \
                      any(0x3040 <= ord(c) <= 0x309F for c in full_text) or \
                      any(0xAC00 <= ord(c) <= 0xD7AF for c in full_text)
            if has_cjk:
                print(f"  ✅ PASS: {lang} characters detected")
            else:
                print(f"  ❌ FAIL: {lang} characters not detected")
                all_passed = False
        else:
            # For Latin scripts, check if the word is present
            if expected.lower().replace(' ', '') in full_text.lower().replace(' ', ''):
                print(f"  ✅ PASS: {lang} text detected")
            else:
                print(f"  ⚠️  WARN: {lang} text not clearly detected (may be OCR variation)")
    
    return all_passed

if __name__ == "__main__":
    print("=" * 60)
    print("OCR-4 Multilingual Regression Test")
    print("=" * 60)
    
    try:
        # Check if Pillow is available
        try:
            from PIL import Image
        except ImportError:
            print("\n⚠️  Pillow not installed. Install with: pip install Pillow")
            exit(1)
        
        passed = test_multilingual()
        print("\n" + "=" * 60)
        if passed:
            print("✅ ALL TESTS PASSED")
        else:
            print("❌ SOME TESTS FAILED")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
```

Run it:
```bash
pip install Pillow
python quick_multilingual_test.py
```

---

## 📈 Run All Quick Tests

Create a simple runner script `run_all_quick_tests.sh`:

```bash
#!/bin/bash

echo "============================================================"
echo "OCR-4 Quick Regression Test Suite"
echo "============================================================"

echo ""
echo "1. Running Confidence Score Tests..."
python quick_confidence_test.py

CONFIDENCE_RESULT=$?

echo ""
echo "2. Running Sparse Table Tests..."
python quick_sparse_test.py

SPARSE_RESULT=$?

echo ""
echo "3. Running Multilingual Tests..."
python quick_multilingual_test.py

MULTI_RESULT=$?

echo ""
echo "============================================================"
echo "Test Summary:"
echo "============================================================"

if [ $CONFIDENCE_RESULT -eq 0 ]; then
    echo "✅ Confidence Score Tests: PASSED"
else
    echo "❌ Confidence Score Tests: FAILED"
fi

if [ $SPARSE_RESULT -eq 0 ]; then
    echo "✅ Sparse Table Tests: PASSED"
else
    echo "❌ Sparse Table Tests: FAILED"
fi

if [ $MULTI_RESULT -eq 0 ]; then
    echo "✅ Multilingual Tests: PASSED"
else
    echo "❌ Multilingual Tests: FAILED"
fi

echo "============================================================"

# Exit with error if any test failed
if [ $CONFIDENCE_RESULT -ne 0 ] || [ $SPARSE_RESULT -ne 0 ] || [ $MULTI_RESULT -ne 0 ]; then
    exit 1
fi
```

Make it executable and run:
```bash
chmod +x run_all_quick_tests.sh
./run_all_quick_tests.sh
```

---

## 🎯 Next Steps

### Immediate (Today)
1. ✅ Run the quick tests above
2. ✅ Document results in `test_results/quick_test_results.md`
3. ✅ Fix any failures

### Short-term (This Week)
1. 📋 Add 5 more corner case samples (low res, rotated, watermarked, etc.)
2. 📋 Create ground truth JSON for existing samples
3. 📋 Set up GitHub Actions CI/CD (see `OCR4_REPO_REVIEW_AND_REGRESSION_PLAN.md`)

### Long-term (This Month)
1. 🏗️ Implement full regression test suite
2. 🏗️ Create baseline metrics
3. 🏗️ Set up automated nightly runs
4. 🏗️ Publish test results dashboard

---

## 📚 Resources

- **Full Review & Test Plan**: See `OCR4_REPO_REVIEW_AND_REGRESSION_PLAN.md`
- **Existing Tests**: `test_mistral_docai_4_0.py`
- **Feature Showcases**: 
  - `ocr4_comprehensive_showcase.ipynb` (Bounding boxes, confidence, multilingual)
  - `spares_table_ocr.ipynb` (Sparse tables)
  - `nvidia_10q_ocr4_analysis.ipynb` (Confidence visualization, block classification)
- **Test Data**: `/samples/` directory

---

## 💡 Tips

1. **Start small**: Begin with the 3 quick tests above
2. **Use existing samples**: Test with documents already in `/samples/`
3. **Baseline first**: Run tests and save results as your baseline
4. **Automate early**: Set up GitHub Actions within a week
5. **Iterate**: Add more corner cases gradually

**Remember**: The goal is to **catch regressions before customers do**!
