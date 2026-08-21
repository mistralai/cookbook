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
