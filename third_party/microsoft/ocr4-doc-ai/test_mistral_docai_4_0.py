#!/usr/bin/env python3
"""
Pytest test cases for Mistral Document AI notebook functions
"""

from asyncio import subprocess
import base64
from email.mime import image
import json
import os
from unittest import result
import requests
from urllib3.util import response
from dotenv import load_dotenv
import pytest
import subprocess
from typing import Dict, Any
import tempfile


load_dotenv()

AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT = os.getenv("AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT")
AZURE_MISTRAL_DOCUMENT_AI_KEY = os.getenv("AZURE_MISTRAL_DOCUMENT_AI_KEY")
AZURE_AI_DEPLOYMENT_NAME = os.getenv("AZURE_AI_DEPLOYMENT_NAME")

REQUEST_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AZURE_MISTRAL_DOCUMENT_AI_KEY}",
}

base_dir = os.path.dirname(os.path.abspath(__file__))

# Import the functions from the notebook
# Note: In a real scenario, these would be imported from a proper Python module
# For testing purposes, we'll redefine them here

def encode_file(file_path: str) -> str:
    """Base64 encode a file for API requests"""
    try:
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None


def bboxannotation(encoding: str, format: str) -> Dict[str, Any]:
    """Create bbox annotation payload"""
    
    bboxannotationPayload = {
        "model": f"{AZURE_AI_DEPLOYMENT_NAME}",
        "document": {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{encoding}",
        },
        "extract_footer": True,
        "extract_header": True, 
        "include_image_base64": "true",
        f"{format}": {
            "type": "json_schema",
            "json_schema": {
                "name": "string",
                "description": "string",
                "schema": {
                    "properties": {
                        "image_type": {
                            "description": "The type of the image.",
                            "title": "Image Type",
                            "type": "string",
                        },
                        "short_description": {
                            "description": "A description in english describing the image.",
                            "title": "Short Description",
                            "type": "string",
                        },
                    }
                },
            },
        },
    }
    return bboxannotationPayload


def parse_bbox_annotation_output(response: Dict[str, Any]):
    """Parse and display bbox annotation output"""
    result = []
    for page in response["pages"]:
        for image in page["images"]:
            result.append(f"page {page['index']}")
            iaj = json.loads(image["image_annotation"])
            result.append(f"Image type: {iaj['properties']['image_type']}")
            result.append(f"Short description: {iaj['properties']['short_description']}")
    return result


class TestEncodeFile:
    """Test cases for encode_file function"""
    
    def test_encode_existing_file(self):
        """Test encoding an existing file"""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name
        
        try:
            # Test encoding
            encoded = encode_file(tmp_path)
            assert encoded is not None
            assert isinstance(encoded, str)
            
            # Verify it's valid base64
            decoded = base64.b64decode(encoded)
            assert decoded == b"test content"
        finally:
            os.unlink(tmp_path)
    
    def test_encode_nonexistent_file(self):
        """Test encoding a non-existent file"""
        result = encode_file("/nonexistent/file.txt")
        assert result is None
    
    def test_encode_pdf_file(self):
        """Test encoding a PDF file from samples"""
        pdf_path = os.path.join(base_dir, "samples", "mistral7b.pdf")
        if os.path.exists(pdf_path):
            encoded = encode_file(pdf_path)
            assert encoded is not None
            assert len(encoded) > 100  # PDF should be reasonably large
        else:
            pytest.skip("Sample PDF file not found")


class TestBBoxAnnotation:
    """Test cases for bboxannotation function"""
    
    def test_bboxannotation_structure(self):
        """Test that bboxannotation returns correct structure"""
        test_encoding = "dGVzdCBlbmNvZGluZw=="  # "test encoding" in base64
        
        result = bboxannotation(test_encoding, "bbox_annotation_format")
        
        # Verify structure
        assert "model" in result
        assert "document" in result
        assert "extract_footer" in result
        assert "extract_header" in result
        assert "include_image_base64" in result
        assert "bbox_annotation_format" in result
        
        # Verify values
        assert result["model"] == AZURE_AI_DEPLOYMENT_NAME
        assert result["extract_footer"] is True
        assert result["extract_header"] is True
        assert result["include_image_base64"] == "true"
        
        # Verify document structure
        assert result["document"]["type"] == "document_url"
        assert test_encoding in result["document"]["document_url"]
    
    def test_bboxannotation_document_annotation_format(self):
        """Test bboxannotation with document_annotation_format"""
        test_encoding = "dGVzdCBlbmNvZGluZw=="
        
        result = bboxannotation(test_encoding, "document_annotation_format")
        
        assert "document_annotation_format" in result
        assert result["document_annotation_format"]["type"] == "json_schema"


class TestBBoxAnnotationOutput:
    """Test cases for parse_bbox_annotation_output function"""
    
    def test_parse_annotation_output(self):
        """Test parsing of annotation output"""
        # Create mock response data
        mock_response = {
            "pages": [
                {
                    "index": 0,
                    "images": [
                        {
                            "image_annotation": json.dumps({
                                "properties": {
                                    "image_type": "Logo",
                                    "short_description": "A company logo"
                                }
                            })
                        }
                    ]
                },
                {
                    "index": 1,
                    "images": [
                        {
                            "image_annotation": json.dumps({
                                "properties": {
                                    "image_type": "Chart",
                                    "short_description": "A bar chart showing sales data"
                                }
                            })
                        }
                    ]
                }
            ]
        }
        
        result = parse_bbox_annotation_output(mock_response)
        
        # Verify output contains expected elements
        assert len(result) == 6  # 3 items per image * 2 images
        assert "page 0" in result[0]
        assert "Image type: Logo" in result[1]
        assert "Short description: A company logo" in result[2]
        assert "page 1" in result[3]
        assert "Image type: Chart" in result[4]
        assert "Short description: A bar chart showing sales data" in result[5]
    
    def test_empty_pages(self):
        """Test with empty pages"""
        mock_response = {"pages": []}
        result = parse_bbox_annotation_output(mock_response)
        assert result == []
    
    def test_pages_without_images(self):
        """Test with pages that have no images"""
        mock_response = {
            "pages": [
                {
                    "index": 0,
                    "images": []
                }
            ]
        }
        result = parse_bbox_annotation_output(mock_response)
        assert result == []


class TestIntegration:
    """Integration tests combining multiple functions"""
    
    def test_encode_and_bboxannotation(self):
        """Test encoding a file and creating bbox annotation payload"""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(b"%PDF-1.4 test pdf content")
            tmp_path = tmp.name
        
        try:
            # Encode the file
            encoded = encode_file(tmp_path)
            assert encoded is not None
            
            # Create bbox annotation payload
            payload = bboxannotation(encoded, "bbox_annotation_format")
            assert payload is not None
            assert "document" in payload
            assert encoded in payload["document"]["document_url"]
            
        finally:
            os.unlink(tmp_path)

    def test_doc_annotation_and_parse_image_output(self):
        """Test creating document annotation payload and parsing document mock output"""
        img_path = os.path.join(base_dir, "samples", "receipt.png")
        bb1Response_image = requests.post(
            url=AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT,
                json=bboxannotation(encode_file(img_path), 'document_annotation_format'),
                headers=REQUEST_HEADERS,
            )       
        response = bb1Response_image.json()
        doc_annotation = json.loads(response["document_annotation"])
        assert "Civic Center" in doc_annotation["properties"]["short_description"]

    def test_bboxannotation_and_parse_pdf_output(self):
        """Test creating bbox annotation payload and parsing pdf bbox mock output"""
        pdf_path = os.path.join(base_dir, "samples", "mistral7b.pdf")
        bb1Response_document = requests.post(
            url=AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT,
            json=bboxannotation(encode_file(pdf_path), 'bbox_annotation_format'),
            headers=REQUEST_HEADERS,
        )     
        image = bb1Response_document.json()["pages"][0]["images"][0]
        iaj = json.loads(image["image_annotation"])
        assert "Mistral AI" in iaj["properties"]["short_description"]
        assert "Mistral 7b" in bb1Response_document.json()["pages"][6]["header"]

    def test_Tabular_data(self):
        """Test Next we look at a document with tabular data, for this example we are using Microsoft's 8-K filing located here: https://www.microsoft.com/en-us/investor/sec-filings """
        ms8kDocument = encode_file(os.path.join(base_dir, "samples", "0000950170-25-100226.pdf"))
        msRequestPayload = {
            "model": f"{AZURE_AI_DEPLOYMENT_NAME}",
            "document": {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{ms8kDocument}",
            },
            "table_format": "html",
            "extract_header": True, 
            "extract_footer": True,
        }
        ms8kResponse = requests.post(
            url=AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT,
            json=msRequestPayload,
            headers=REQUEST_HEADERS,
        )
        page_data = ms8kResponse.json()["pages"][0]
        assert "SECURITIES AND EXCHANGE COMMISSION" in (page_data["header"] or page_data.get("markdown", ""))
        assert "Title of each class" in page_data["tables"][0]['content']
        assert "Common stock, $0.00000625 par value" in page_data["tables"][0]['content']

    def test_Word_Document(self):
        """Test processing a Word document"""
        word_doc_path = os.path.join(base_dir, "samples", "TranscriptFY25q4.docx")
        if os.path.exists(word_doc_path):
            encoded_doc = encode_file(word_doc_path)
            wordPayload = {
                "model": f"{AZURE_AI_DEPLOYMENT_NAME}",
                "document": {
                    "type": "document_url",
                    "document_url": f"data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{encoded_doc}",
                },
                "table_format": "html",
                "extract_header": True, 
            "extract_footer": True, 
            }
            wordResponse = requests.post(
                url=AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT,
                json=wordPayload,
                headers=REQUEST_HEADERS,
            )
            assert wordResponse.status_code == 200
            assert "Cloud and AI is the driving force of business transformation across every industry" in wordResponse.json()["pages"][0]['markdown']
        else:
            pytest.skip("Sample Word document not found")


        def test_powerpoint(self):
            """Test processing a PowerPoint document"""
            ppt_doc_path = os.path.join(base_dir, "samples", "sample.pptx")
            if os.path.exists(ppt_doc_path):
                encoded_doc = encode_file(ppt_doc_path)
                pptPayload = {
                    "model": f"{AZURE_AI_DEPLOYMENT_NAME}",
                        "document": {
                            "type": "document_url",
                            "document_url": f"data:application/vnd.openxmlformats-officedocument.presentationml.presentation;base64,{encoded_doc}",
                        },
                        "include_image_base64": "false",
                        "image_limit": 0,
                        "table_format": "html",
                        "extract_header": True, 
                        "extract_footer": True, 
                    }
                pptResponse = requests.post(
                    url=AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT,
                    json=pptPayload,
                    headers=REQUEST_HEADERS,
                )
                assert pptResponse.status_code == 200
                assert "onfigurable AI for all builders" in pptResponse.json()["pages"][0]['markdown']
            else:
                pytest.skip("Sample PowerPoint document not found")

    def test_epub_document(self):
        """Test processing an EPUB document"""
        epub_doc_path = os.path.join(base_dir, "samples", "minimal.epub")
        if os.path.exists(epub_doc_path):
            encoded_doc = encode_file(epub_doc_path)
            epubPayload = {
                "model": f"{AZURE_AI_DEPLOYMENT_NAME}",
                "document": {
                    "type": "document_url",
                    "document_url": f"data:application/epub+zip;base64,{encoded_doc}",
                },
                "include_image_base64": "false",
                "image_limit": 0,
                "table_format": "html",
                "extract_header": True, 
                "extract_footer": True, 
            }
            epubResponse = requests.post(
                url=AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT,
                json=epubPayload,
                headers=REQUEST_HEADERS,
            )
            assert epubResponse.status_code == 200
            assert "Integer ultricies nisi nec nisi gravida" in epubResponse.json()["pages"][0]['markdown']
        else:
            pytest.skip("Sample EPUB document not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])