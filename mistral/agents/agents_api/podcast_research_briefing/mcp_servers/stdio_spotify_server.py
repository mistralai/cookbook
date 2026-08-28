from mcp.server.fastmcp import FastMCP
import logging
import os
import json

# Configure logging to only show errors
logging.basicConfig(level=logging.ERROR)

# Initialize FastMCP server for Spotify podcast search
mcp = FastMCP("spotify_podcasts")

_spotify_client = None


def _get_spotify_client():
    """Lazy initialization of the Spotify client using Client Credentials auth."""
    global _spotify_client
    if _spotify_client is None:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials

        auth_manager = SpotifyClientCredentials(
            client_id=os.environ["SPOTIFY_CLIENT_ID"],
            client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
        )
        _spotify_client = spotipy.Spotify(auth_manager=auth_manager)
    return _spotify_client


def _format_duration(ms: int) -> str:
    """Convert milliseconds to a human-readable duration string."""
    minutes = ms // 60000
    if minutes >= 60:
        hours = minutes // 60
        remaining = minutes % 60
        return f"{hours}h {remaining}m"
    return f"{minutes}m"


@mcp.tool()
async def search_podcasts(query: str, limit: int = 10) -> str:
    """
    Search for podcast shows on Spotify matching a topic or keyword.

    Args:
        query (str): Search query for finding podcast shows.
        limit (int): Maximum number of results to return. Defaults to 10.

    Returns:
        str: JSON array of matching podcast shows with name, publisher,
             description, total episodes, and Spotify URL.
    """
    try:
        sp = _get_spotify_client()
        results = sp.search(q=query, type="show", limit=limit)
        shows = []
        for item in results.get("shows", {}).get("items", []):
            if item is None:
                continue
            shows.append({
                "id": item["id"],
                "name": item["name"],
                "publisher": item.get("publisher", "Unknown"),
                "description": (item.get("description") or "")[:500],
                "total_episodes": item.get("total_episodes", 0),
                "url": item.get("external_urls", {}).get("spotify", ""),
            })
        return json.dumps(shows, indent=2)
    except Exception as e:
        return f"Error searching podcasts: {e}"


@mcp.tool()
async def search_episodes(query: str, limit: int = 10) -> str:
    """
    Search for specific podcast episodes on Spotify matching a topic or keyword.

    Args:
        query (str): Search query for finding podcast episodes.
        limit (int): Maximum number of results to return. Defaults to 10.

    Returns:
        str: JSON array of matching episodes with name, show name,
             description, duration, release date, and Spotify URL.
    """
    try:
        sp = _get_spotify_client()
        results = sp.search(q=query, type="episode", limit=limit)
        episodes = []
        for item in results.get("episodes", {}).get("items", []):
            if item is None:
                continue
            episodes.append({
                "id": item["id"],
                "name": item["name"],
                "show_name": item.get("show", {}).get("name", "Unknown"),
                "description": (item.get("description") or "")[:500],
                "duration": _format_duration(item.get("duration_ms", 0)),
                "release_date": item.get("release_date", "Unknown"),
                "url": item.get("external_urls", {}).get("spotify", ""),
            })
        return json.dumps(episodes, indent=2)
    except Exception as e:
        return f"Error searching episodes: {e}"


@mcp.tool()
async def get_podcast_details(show_id: str) -> str:
    """
    Get full details for a specific podcast show by its Spotify ID.

    Args:
        show_id (str): The Spotify show ID.

    Returns:
        str: JSON object with show name, publisher, description,
             total episodes, languages, and Spotify URL.
    """
    try:
        sp = _get_spotify_client()
        show = sp.show(show_id)
        return json.dumps({
            "id": show["id"],
            "name": show["name"],
            "publisher": show.get("publisher", "Unknown"),
            "description": (show.get("description") or "")[:1000],
            "total_episodes": show.get("total_episodes", 0),
            "languages": show.get("languages", []),
            "url": show.get("external_urls", {}).get("spotify", ""),
        }, indent=2)
    except Exception as e:
        return f"Error fetching podcast details for {show_id}: {e}"


@mcp.tool()
async def get_podcast_episodes(show_id: str, limit: int = 10) -> str:
    """
    Get episodes from a specific podcast show.

    Args:
        show_id (str): The Spotify show ID.
        limit (int): Maximum number of episodes to return. Defaults to 10.

    Returns:
        str: JSON array of episodes with name, description, duration,
             release date, and Spotify URL.
    """
    try:
        sp = _get_spotify_client()
        results = sp.show_episodes(show_id, limit=limit)
        episodes = []
        for item in results.get("items", []):
            if item is None:
                continue
            episodes.append({
                "id": item["id"],
                "name": item["name"],
                "description": (item.get("description") or "")[:500],
                "duration": _format_duration(item.get("duration_ms", 0)),
                "release_date": item.get("release_date", "Unknown"),
                "url": item.get("external_urls", {}).get("spotify", ""),
            })
        return json.dumps(episodes, indent=2)
    except Exception as e:
        return f"Error fetching episodes for show {show_id}: {e}"


@mcp.tool()
async def get_episode_details(episode_id: str) -> str:
    """
    Get full details for a specific podcast episode by its Spotify ID.

    Args:
        episode_id (str): The Spotify episode ID.

    Returns:
        str: JSON object with episode name, show name, full description,
             duration, release date, and Spotify URL.
    """
    try:
        sp = _get_spotify_client()
        episode = sp.episode(episode_id)
        return json.dumps({
            "id": episode["id"],
            "name": episode["name"],
            "show_name": episode.get("show", {}).get("name", "Unknown"),
            "description": (episode.get("description") or "")[:2000],
            "duration": _format_duration(episode.get("duration_ms", 0)),
            "release_date": episode.get("release_date", "Unknown"),
            "language": episode.get("language", "Unknown"),
            "url": episode.get("external_urls", {}).get("spotify", ""),
        }, indent=2)
    except Exception as e:
        return f"Error fetching episode details for {episode_id}: {e}"


def run_spotify_server():
    """Start the Spotify podcast MCP server using stdio transport"""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_spotify_server()
