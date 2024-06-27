import json
import os
import requests
from indexify import IndexifyClient

def download_pdf(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"PDF downloaded and saved to {save_path}")


def extract_entities_from_pdf(pdf_path):
    client = IndexifyClient()
    
    # Upload the PDF file
    content_id = client.upload_file("pdf_entity_extractor", pdf_path)
    
    # Wait for the extraction to complete
    client.wait_for_extraction(content_id)
    
    # Retrieve the extracted entities
    entities_content = client.get_extracted_content(
        content_id=content_id,
        graph_name="pdf_entity_extractor",
        policy_name="text_to_entities"
    )
    
    # Parse the JSON response
    entities = json.loads(entities_content[0]['content'].decode('utf-8'))
    return entities

# Example usage
if __name__ == "__main__":
    pdf_url = "https://arxiv.org/pdf/2310.06825.pdf"
    pdf_path = "reference_document.pdf"

    # Download the PDF
    download_pdf(pdf_url, pdf_path)
    extracted_entities = extract_entities_from_pdf(pdf_path)
    
    print("Extracted Entities:")
    for category, entities in extracted_entities.items():
        print(f"\n{category.capitalize()}:")
        for entity in entities:
            print(f"- {entity}")