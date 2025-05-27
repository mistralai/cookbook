from mcp.server.fastmcp import FastMCP
import logging
import os
from typing import List
from mistralai import Mistral

# Configure logging to only show errors
logging.basicConfig(level=logging.ERROR)

# Initialize FastMCP server for report generation
mcp = FastMCP("report_generator")

# Initialize MistralAI client for report content generation
client = Mistral(api_key="<YOUR MISTRALAI API KEY>")

# System prompt to guide the LLM in generating financial reports
system_prompt = "You are a professional financial analyst. Generate a very short report based on following information regarding different companies."

@mcp.tool()
def generate_report_content(prompt: str) -> str:
    """
    Generate financial report content using MistralAI LLM based on provided data
    
    Args:
        prompt (str): The prompt containing financial data and analysis requirements
    
    Returns:
        str: Generated report content or error message
    """
    try:
        response = client.chat.complete(
            model="mistral-medium-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating report content: {str(e)}"

def run_report_server():
    """Start the report generation MCP server using stdio transport"""
    mcp.run(transport="stdio")

if __name__ == "__main__":
    run_report_server()