import os
import requests
from indexify import IndexifyClient

def download_pdf(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"PDF downloaded and saved to {save_path}")

def summarize_pdf(pdf_path):
    client = IndexifyClient()
    
    # Upload the PDF file
    content_id = client.upload_file("pdf_summarizer", pdf_path)
    
    # Wait for the extraction to complete
    client.wait_for_extraction(content_id)
    
    # Retrieve the summarized content
    summary = client.get_extracted_content(
        content_id=content_id,
        graph_name="pdf_summarizer",
        policy_name="text_to_summary"
    )
    
    return summary[0]['content'].decode('utf-8')

# Example usage
if __name__ == "__main__":
    pdf_url = "https://arxiv.org/pdf/2310.06825.pdf"
    pdf_path = "reference_document.pdf"
    
    # Download the PDF
    download_pdf(pdf_url, pdf_path)
    
    # Summarize the PDF
    summary = summarize_pdf(pdf_path)
    print("Summary of the PDF:")
    print(summary)