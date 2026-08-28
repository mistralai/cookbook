from mcp.server.fastmcp import FastMCP
import logging
import os
from mistralai.client import Mistral

# Configure logging to only show errors
logging.basicConfig(level=logging.ERROR)

# Initialize FastMCP server for briefing generation
mcp = FastMCP("briefing_generator")

# Initialize Mistral client
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# System prompt for generating structured podcast research briefings
system_prompt = """You are a research analyst specializing in podcast content curation.
Given a topic, podcast/episode data from Spotify, and supplementary web research,
produce a structured research briefing in markdown format with the following sections:

## Executive Summary
A concise overview of the podcast landscape for this topic (2-3 sentences).

## Ranked Episode Recommendations
A numbered list of the most relevant episodes, each with:
- **Relevance Score** (1-10)
- **Episode Name** and **Show Name**
- **Spotify Link** — use ONLY the exact URL from the "url" field in the input data. NEVER fabricate or guess Spotify URLs. If no URL is provided for an episode, write "Link not available" instead.
- **Duration** and **Release Date**
- A 2-3 sentence summary of why this episode is relevant

Rank by relevance to the research topic, recency, and quality of the source.

CRITICAL: Every Spotify link you include MUST be copied verbatim from the input data.
Do not construct URLs yourself. Spotify URLs follow the pattern
https://open.spotify.com/episode/... — if a URL in your output does not appear
in the input data, remove it.

## Key Themes Across Episodes
Identify 3-5 recurring themes or perspectives found across the recommended episodes.

## Notable Experts & Guests
List any notable guests, hosts, or experts mentioned in the episode descriptions,
with brief context on their relevance.

## Suggested Deep Dives
Recommend 2-3 specific follow-up research directions based on gaps or
interesting threads found in the podcast content.

## Gaps & Limitations
Note any limitations in the available podcast content for this topic,
such as missing perspectives, outdated information, or geographic bias."""


@mcp.tool()
def generate_research_briefing(topic: str, podcast_data: str, web_research: str) -> str:
    """
    Generate a structured research briefing from podcast data and web research.

    Args:
        topic (str): The research topic being investigated.
        podcast_data (str): JSON string of podcast and episode data from Spotify.
        web_research (str): Additional context gathered from web search.

    Returns:
        str: A structured markdown briefing with ranked recommendations and analysis.
    """
    try:
        user_prompt = f"""Research Topic: {topic}

Podcast & Episode Data from Spotify:
{podcast_data}

Supplementary Web Research:
{web_research}

Generate a comprehensive research briefing based on the above information.
For every episode you recommend, copy the exact Spotify URL from the input data above.
Never generate or guess a URL — only use URLs that appear verbatim in the podcast data."""

        response = client.chat.complete(
            model="mistral-medium-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating research briefing: {str(e)}"


def run_briefing_server():
    """Start the briefing generation MCP server using stdio transport"""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_briefing_server()
