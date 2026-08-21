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
