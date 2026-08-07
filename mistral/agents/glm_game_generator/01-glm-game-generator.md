# Generate a playable mini game with GLM

Use the `zai-glm-5-2` (GLM) model through the Mistral API to generate a complete, playable HTML5 dungeon crawler from a single prompt.

---

## Prerequisites

To complete this cookbook, you will need:
- Python 3.10+
- A Mistral account and API key

## Environment setup

### Install

Install the Mistral Python SDK and `python-dotenv` for loading your API key from a `.env` file:

```bash
pip install mistralai python-dotenv
```

### Required environment variables

To complete this cookbook, you'll need a Mistral API key. In [Studio](https://console.mistral.ai), navigate to the [API keys section](https://console.mistral.ai/home?profile_dialog=api-keys) and create a new API key.

Create a `.env` at the root of your project and add your Mistral API key:

```
MISTRAL_API_KEY=your-mistral-api-key
```

---

## Step 1 — Initialize the client

Create `generate_game.py` in your project directory:

```bash
touch generate_game.py
```

Open the file and add the imports and client initialization. The remaining steps build out the prompt, the API call, and the local server.

```python
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


def main():
    # Step 2 — Define the game description
    # Step 3 — Craft the prompt
    # Step 4 — Call the model and extract HTML
    # Step 5 — Serve the game locally
    pass


if __name__ == "__main__":
    main()
```

---

## Step 2 — Define the game description

Describe the game you want to generate. Be specific about mechanics, controls, visuals, and scope — the more detail you provide, the better the result. This description produces a top-down dungeon crawler with procedural rooms, enemies, items, and a minimap.

Add the following above the `main` function:

```python
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
```

---

## Step 3 — Craft the prompt

The prompt has two parts. The system message constrains the output format — single HTML file, no external dependencies, Canvas rendering. The user message describes the game and lists concrete requirements so the model doesn't omit features like a start screen or game-over logic.

Add the following function above `main`:

```python
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
```

---

## Step 4 — Call the model and extract HTML

Send the prompt to GLM and extract the HTML from the response. Two things to note:

- **Timeout**: `timeout_ms=600_000` gives the model 10 minutes. GLM generates large code outputs (1000+ lines), so this headroom prevents the request from timing out.
- **Extraction**: The model wraps its output in a ` ```html ` code fence. The `extract_html` function parses this, falling back to DOCTYPE-based extraction if no fence is found.

Add the `extract_html` function above `main`:

```python
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
```

Replace `# Step 3 — Craft the prompt` and `# Step 4 — Call the model and extract HTML` in `main` with:

```python
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
```

---

## Step 5 — Serve the game locally

Opening `file://` URLs in a browser triggers security restrictions that can break JavaScript execution. A local HTTP server avoids this entirely.

Add the `serve_and_open` function above `main`:

```python
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
```

Replace `# Step 5 — Serve the game locally` in `main` with:

```python
    # Step 5 — Serve the game locally
    serve_and_open(output.resolve().parent, output.name)
```

---

## Run

Once all steps are in place, run the script:

```bash
python generate_game.py
```

The script calls GLM, saves the generated HTML to `game.html`, starts a local server at `http://localhost:8000/game.html`, and opens it in your browser. Press Ctrl+C to stop the server.

Example output:

```
Generating game: A top-down dungeon crawler. The player navigates procedurally generated rooms...
This may take a few minutes...
Game saved to /Users/you/glm_game_generator/game.html
Serving game at http://localhost:8000/game.html
Press Ctrl+C to stop the server.
```

---

## Try different games

Change `GAME_DESCRIPTION` to generate a different game. Here are a few ideas:

**Space shooter:**

```python
GAME_DESCRIPTION = (
    "A vertical-scrolling space shooter. The player controls a ship at the bottom "
    "of the screen, moves left/right with arrow keys, and shoots with spacebar. "
    "Waves of enemy ships descend from the top with different movement patterns. "
    "Power-ups drop from destroyed enemies: rapid fire, shield, triple shot. "
    "Track score and display a high-score counter."
)
```

**Breakout clone:**

```python
GAME_DESCRIPTION = (
    "A Breakout/Arkanoid clone. The player controls a paddle at the bottom with "
    "left/right arrow keys. A ball bounces around the screen destroying colored "
    "bricks. Different brick colors take different numbers of hits. Some bricks "
    "drop power-ups: wider paddle, multi-ball, sticky paddle. Include 3 levels "
    "with different brick layouts."
)
```

**Tower defense:**

```python
GAME_DESCRIPTION = (
    "A tower defense game. Enemies follow a winding path from the top-left to "
    "the bottom-right. Click on empty tiles adjacent to the path to place towers. "
    "Three tower types: arrow (fast, low damage), cannon (slow, splash damage), "
    "and ice (slows enemies). Earn gold from defeated enemies to buy more towers. "
    "Survive 10 waves with increasing difficulty."
)
```

---

## Complete script

For reference, here is the full script with all steps combined:

```python
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
```

---

## Summary

This cookbook demonstrated how to use GLM as a code-generation engine — send a detailed game description, extract the HTML output, and serve it locally for instant play.

**What you built:**
- A game generation script that turns a text description into a playable HTML5 Canvas game
- A prompt structure that constrains output format (single file, no dependencies) while describing game mechanics in detail
- A local HTTP server that serves the generated game without browser security restrictions

**Mistral features used:**
- Chat completions API with the `zai-glm-5-2` model
- System and user message roles for structured prompting
- Extended timeout (`timeout_ms`) for long code generation

Try describing your own game idea and see what GLM produces. For more on available models, see the [models documentation](https://docs.mistral.ai/getting-started/models/models_overview/).
