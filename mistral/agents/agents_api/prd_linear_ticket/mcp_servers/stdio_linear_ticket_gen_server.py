from mcp.server.fastmcp import FastMCP
import logging
import json
from datetime import datetime
import os
from typing import List, Dict, Any
from pydantic import BaseModel
from mistralai import Mistral
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

# Configure logging to only show errors
logging.basicConfig(level=logging.ERROR)

# Initialize FastMCP server for Linear ticket generation
mcp = FastMCP("linear_ticket_generator")

# Initialize MistralAI client for PRD parsing
mistral_client = Mistral(api_key="<YOUR MISTRALAI API KEY>")

# Linear API configuration
linear_api_key = "<YOUR LINEAR API KEY>"
graphql_url = "https://api.linear.app/graphql"
team_id = "<YOUR LINEAR TEAM ID>"

# Initialize GraphQL client for Linear API
client = Client(
    transport=RequestsHTTPTransport(
        url=graphql_url,
        headers={'Authorization': linear_api_key},
        verify=True,
        retries=3
    ),
    fetch_schema_from_transport=True
)

class FeaturesList(BaseModel):
    """Pydantic model for structured feature data from PRD parsing"""
    Features: List[str]
    DescriptionOfFeatures: List[str]

def parse_prd(prd_text: str) -> Dict[str, List[str]]:
    """
    Parse PRD text into structured feature data using MistralAI

    Args:
        prd_text (str): PRD text to parse

    Returns:
        Dict[str, List[str]]: Structured feature data with titles and descriptions
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant helping to create Features list and their descriptions from a Product Requirements Document (PRD)."
                "The description should contain a brief explanation of the feature that includes Technical requirements (if any), Constraints (if any), Success metrics (if any), User personas (if any), and Timeline and Milestones (if any)."
            )
        },
        {
            "role": "user",
            "content": f"PRD:\n\n{prd_text}"
        }
    ]

    # Use MistralAI to parse PRD into structured format
    chat_response = mistral_client.chat.parse(
        model="mistral-medium-latest",
        messages=messages,
        response_format=FeaturesList,
        max_tokens=2048,
        temperature=0.1
    )

    return json.loads(chat_response.choices[0].message.content)

def create_ticket(title: str, description: str) -> Dict[str, Any]:
    """
    Create a single Linear ticket using GraphQL API

    Args:
        title (str): Ticket title
        description (str): Ticket description

    Returns:
        Dict[str, Any]: Creation result from Linear API
    """
    # GraphQL mutation to create a new issue in Linear
    mutation = gql("""
    mutation CreateIssue($title: String!, $description: String!, $teamId: String!) {
        issueCreate(
            input: {
                title: $title,
                description: $description,
                teamId: $teamId
            }
        ) {
            success
            issue {
                id
                url
            }
        }
    }
    """)

    variables = {
        "title": title,
        "description": description,
        "teamId": team_id
    }

    # Execute the mutation and return result
    result = client.execute(mutation, variable_values=variables)
    print(f"Created ticket: {result['issueCreate']['issue']['url']}")
    return result

@mcp.tool()
def create_tickets_from_prd(prd: str) -> List[Dict[str, Any]]:
    """
    Parse PRD and create corresponding Linear tickets for each feature

    Args:
        prd (str): Product Requirements Document text

    Returns:
        List[Dict[str, Any]]: List of ticket creation results from Linear API
    """
    # Parse the PRD into structured feature data
    parsed_items = parse_prd(prd)
    results = []
    
    # Create a Linear ticket for each feature
    for title, description in zip(
        parsed_items['Features'],
        parsed_items['DescriptionOfFeatures']
    ):
        result = create_ticket(title, description)
        results.append(result)
    return results

def run_linear_ticket_gen_server():
    """Start the Linear ticket generation MCP server using stdio transport"""
    mcp.run(transport="stdio")

if __name__ == "__main__":
    run_linear_ticket_gen_server()