"""Generate a playable HTML5 mini game using the GLM model via the Mistral API."""

import functools
import http.server
import os
import re
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

from mistralai.client import Mistral

load_dotenv()

# Step 1 — Initialize the client
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# Step 2 — Define the game description
# Be specific about mechanics, controls, visuals, and scope.
# The more detail you provide, the better the generated game.
GAME_DESCRIPTION = (
    "A top-down dungeon crawler. The player navigates procedurally generated "
    "rooms connected by doorways. Each room contains enemies that patrol and "
    "chase the player on sight. Defeating enemies drops health potions or score "
    "pickups. The player has a melee attack (spacebar) and 3 lives. Generate at "
    "least 5 connected rooms. Show a minimap in the corner."
)


# Step 3 — Craft the prompt
# The system message constrains the output format (single HTML file, no
# external dependencies, Canvas rendering). The user message describes the
# game and lists concrete requirements so the model doesn't omit features.
def build_game_prompt(game_description: str) -> tuple[str, str]:
    """Build the system and user prompts for game generation."""
    system_prompt = (
        "You are an expert game developer. You produce complete, self-contained "
        "HTML files with embedded CSS and JavaScript. Never use external CDNs, "
        "libraries, or dependencies. Use HTML5 Canvas for rendering. The game "
        "must be fully playable in any modern browser by opening the HTML file directly."
    )
    user_prompt = (
        f"Create a complete, playable game: {game_description}\n\n"
        "Requirements:\n"
        "- Single HTML file with all CSS and JS embedded\n"
        "- No external dependencies, CDNs, or imports\n"
        "- Use HTML5 Canvas for rendering\n"
        "- Keyboard controls (arrow keys or WASD)\n"
        "- Include a start screen with instructions\n"
        "- Track and display score and health/lives\n"
        "- Include game-over and restart logic\n"
        "- Use requestAnimationFrame for the game loop\n"
        "- Add colors, simple shapes, or pixel art for visuals\n\n"
        "Return the complete HTML file inside a single ```html code fence."
    )
    return system_prompt, user_prompt


# Step 4 — Extract HTML from the response
# The model wraps its output in a ```html code fence. This function extracts
# the HTML content, falling back to DOCTYPE-based extraction if no fence is found.
def extract_html(text: str) -> str:
    """Extract HTML content from the model response."""
    # Try fenced code block first
    match = re.search(r"```html\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fall back to DOCTYPE extraction
    match = re.search(r"(<!DOCTYPE.*?</html>)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    raise ValueError("No HTML content found in the model response.")


# Step 5 — Serve the game locally
# Opening file:// URLs triggers browser security restrictions. A local HTTP
# server avoids this and lets the game run without issues.
def serve_and_open(directory: Path, filename: str, port: int = 8000):
    """Start a local HTTP server and open the game in the default browser."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )
    server = http.server.HTTPServer(("localhost", port), handler)
    url = f"http://localhost:{port}/{filename}"

    print(f"Serving game at {url}")
    print("Press Ctrl+C to stop the server.")
    webbrowser.open(url)
    server.serve_forever()


def main():
    # Step 3 — Craft the prompt
    system_prompt, user_prompt = build_game_prompt(GAME_DESCRIPTION)

    # Step 4 — Call the model and extract HTML
    print(f"Generating game: {GAME_DESCRIPTION}")
    print("This may take a few minutes...")

    response = client.chat.complete(
        model="zai-glm-5-2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        timeout_ms=600_000,  # 10-minute timeout for large code generation
    )

    html_content = extract_html(response.choices[0].message.content)

    output = Path("game.html")
    output.write_text(html_content, encoding="utf-8")
    print(f"Game saved to {output.resolve()}")

    # Step 5 — Serve the game locally
    serve_and_open(output.resolve().parent, output.name)


if __name__ == "__main__":
    main()
