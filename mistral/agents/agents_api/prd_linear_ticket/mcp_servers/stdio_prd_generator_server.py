from mcp.server.fastmcp import FastMCP
import logging
from mistralai import Mistral
import traceback
import os

# Configure logging to only show errors
logging.basicConfig(level=logging.ERROR)

# Initialize FastMCP server for PRD generation
mcp = FastMCP("prd_generator")

# Initialize MistralAI client for OCR and PRD generation
mistral_client = Mistral(api_key="<YOUR MISTRALAI API KEY>")

def parse_transcript(file_path: str) -> str:
    """Parse a transcript PDF file and extract text from all pages using MistralAI OCR
    
    Args:
        file_path (str): Path to the PDF file to process
        
    Returns:
        str: Extracted text content from the PDF
    """
    # Upload PDF file to MistralAI for OCR processing
    uploaded_pdf = mistral_client.files.upload(
        file={
            "file_name": file_path,
            "content": open(file_path, "rb"),
        },
        purpose="ocr"
    )

    # Get signed URL for the uploaded file
    signed_url = mistral_client.files.get_signed_url(file_id=uploaded_pdf.id)

    # Process the document using MistralAI OCR
    ocr_response = mistral_client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        }
    )

    # Extract and combine text from all pages
    text = "\n".join([x.markdown for x in (ocr_response.pages)])
    return text

@mcp.tool()
async def generate_prd(file_name: str) -> str:
    """
    Generate Product Requirements Document from a transcript PDF file

    Args:
        file_name (str): Path to the PDF transcript file

    Returns:
        str: Generated PRD text or error message
    """
    try:
        # Extract text from the PDF transcript
        transcript = parse_transcript(file_name)

        # Create prompt for PRD generation
        prompt = f"""
        Based on the following call transcript, create an initial Product Requirements Document (PRD) with some or all of these sections:
        1. Title
        2. Purpose
        3. Scope
        4. Features and Requirements
        5. User Personas
        6. Technical Requirements
        7. Constraints
        8. Success Metrics
        9. Timeline and Milestones

        Transcript:
        {transcript}

        Align everything only with the information provided in the transcript. If any section is not present in the transcript, you can skip it in the PRD.

        PRD:
        """
        
        # Generate PRD using MistralAI
        response = mistral_client.chat.complete(
            model="mistral-medium-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logging.error(f"Error in generate_prd: {str(e)}")
        return traceback.format_exc()

def run_prd_gen_server():
    """Start the PRD generation MCP server using stdio transport"""
    mcp.run(transport="stdio")

if __name__ == "__main__":
    run_prd_gen_server()
