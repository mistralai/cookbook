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
