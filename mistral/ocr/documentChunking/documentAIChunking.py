import argparse
import base64
import logging
import os
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.errors import PdfReadError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def encode_pdf_to_base64(pdf_path: str) -> str:
    """Encode the contents of a PDF file to a base64 string."""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()
            return base64.b64encode(pdf_data).decode('utf-8')
    except FileNotFoundError:
        logger.error(f"File not found: {pdf_path}")
        raise
    except IOError as e:
        logger.error(f"Error reading file {pdf_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error encoding PDF to base64: {e}")
        raise

def count_pdf_pages(pdf_path: str) -> int:
    """Count the number of pages in a PDF file."""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            return len(pdf_reader.pages)
    except FileNotFoundError:
        logger.error(f"File not found: {pdf_path}")
        raise
    except PdfReadError as e:
        logger.error(f"Error reading PDF structure: {e}")
        raise
    except Exception as e:
        logger.error(f"Error counting PDF pages: {e}")
        raise

def get_pdf_size_in_mb(pdf_path: str) -> float:
    """Get the size of a PDF file in MB."""
    try:
        size_in_bytes = os.path.getsize(pdf_path)
        return size_in_bytes / (1024 * 1024)  # Convert to MB
    except FileNotFoundError:
        logger.error(f"File not found: {pdf_path}")
        raise
    except OSError as e:
        logger.error(f"Error getting file size: {e}")
        raise

def split_pdf_by_pages(pdf_path: str, max_pages: int) -> list:
    """Split a PDF into multiple PDFs with a maximum number of pages."""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            pdf_writers = []

            for i in range(0, len(pdf_reader.pages), max_pages):
                pdf_writer = PdfWriter()
                for page_num in range(i, min(i + max_pages, len(pdf_reader.pages))):
                    pdf_writer.add_page(pdf_reader.pages[page_num])
                pdf_writers.append(pdf_writer)

        return pdf_writers
    except FileNotFoundError:
        logger.error(f"File not found: {pdf_path}")
        raise
    except PdfReadError as e:
        logger.error(f"Error reading PDF structure: {e}")
        raise
    except Exception as e:
        logger.error(f"Error splitting PDF: {e}")
        raise



def send_to_api(base64_pdf: str, endpoint: str, api_key: str, model_name: str):
    """Send the base64 encoded PDF to the specified API endpoint."""
    import httpx
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(httpx.HTTPStatusError)
    )
    def _make_request():
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"} if api_key else {}
        timeout = httpx.Timeout(30.0, read=30.0)
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                endpoint,
                json={
                    "model": model_name,
                    "document": {
                            "type": "document_url",
                            "document_url": f"data:application/pdf;base64,{base64_pdf}",
                    }},
                headers=headers
            )
            response.raise_for_status()
            logger.info(f"API request successful: {response.status_code}")
            return response
    try:
        return _make_request()
    except httpx.HTTPStatusError as e:
        logger.error(f"API request failed with status {e.response.status_code}: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error sending request to API: {e}")
        raise


def main() -> None:
    """Splits a single document into smaller parts and calls OCR or Document AI asynchronously."""
    try:
        parser = argparse.ArgumentParser(description='Encode PDF to base64 and split if necessary.')
        parser.add_argument('--pdf_file', type=str, help='Path to the PDF file')
        parser.add_argument('--max_size_mb', type=float, required=True, help='Maximum size of the PDF in MB')
        parser.add_argument('--max_pages', type=int, required=True, help='Maximum number of pages allowed in a single PDF')
        parser.add_argument('--endpoint', type=str, help='Endpoint URL')
        parser.add_argument('--api_key', type=str, help='API key for the endpoint')
        parser.add_argument('--model_name', type=str, help='Model name for the endpoint', default="mistral-ocr-2505")

        args = parser.parse_args()

        pdf_path = args.pdf_file
        max_size_mb = args.max_size_mb
        max_pages = args.max_pages

        # Check if file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return

        # Check PDF size and pages
        try:
            size_in_mb = get_pdf_size_in_mb(pdf_path)
            num_pages = count_pdf_pages(pdf_path)
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return

        if size_in_mb <= max_size_mb and num_pages <= max_pages:
        # Encode the whole PDF if it meets the criteria
            try:
                base64_pdf = encode_pdf_to_base64(pdf_path)
                logger.info("PDF has been encoded to base64 and did not need to be split")
                # TODO: Add logic to send the base64_pdf to the API
                response = send_to_api(base64_pdf, args.endpoint, args.api_key, args.model_name)
                logger.info(f"API response: {response}")
            except Exception as e:
                logger.error(f"Failed to encode PDF: {e}")
        else:
            logger.info("PDF exceeds the size or page limit. Splitting into parts...")
            # Split the PDF into multiple PDFs and encode each one
            try:
                pdf_writers = split_pdf_by_pages(pdf_path, max_pages)

                for i, pdf_writer in enumerate(pdf_writers):
                    split_pdf_path = f'split_part_{i}.pdf'
                    try:
                        with open(split_pdf_path, 'wb') as split_pdf_file:
                            pdf_writer.write(split_pdf_file)
                            base64_pdf = encode_pdf_to_base64(split_pdf_path)
                            logger.info(f"Part {i+1} processed successfully")
                            response = send_to_api(base64_pdf, args.endpoint, args.api_key, args.model_name)
                            logger.info(f"API response: {response}")
                    except Exception as e:
                        logger.error(f"Error processing part {i+1}: {e}")
                    finally:
                        try:
                            if os.path.exists(split_pdf_path):
                                os.remove(split_pdf_path)  # Clean up the split PDF
                        except OSError as e:
                            logger.warning(f"Could not delete temporary file {split_pdf_path}: {e}")

            except Exception as e:
                logger.error(f"Failed to split PDF: {e}")

    except argparse.ArgumentError as e:
        logger.error(f"Argument error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

if __name__ == '__main__':
    main()
