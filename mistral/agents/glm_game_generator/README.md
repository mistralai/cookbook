# Generate a playable mini game with GLM

Use the `zai-glm-5-2` (GLM) model through the Mistral API to generate a complete, playable HTML5 dungeon crawler game from a single prompt.

## Prerequisites

### Install

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Required environment variables

To complete this cookbook, you'll need a Mistral API key. In [Studio](https://console.mistral.ai), navigate to the [API keys section](https://console.mistral.ai/home?profile_dialog=api-keys) and create a new API key.

Create a `.env` at the root of your project and add your Mistral API key:

```
MISTRAL_API_KEY=your-mistral-api-key
```

## 1. Understand the approach

GLM (`zai-glm-5-2`) is a code-generation model available through the Mistral API. In this cookbook, you'll use it to generate a self-contained HTML file that includes all CSS and JavaScript inline. A local HTTP server serves the file so it runs without browser security restrictions.

The workflow is:

1. **Craft a prompt** describing the game you want
2. **Call the model** and extract the HTML from the response
3. **Serve the file** on localhost and open it in your browser

The generated game uses HTML5 Canvas for rendering and `requestAnimationFrame` for the game loop, so it runs at full frame rate in any modern browser.

## 2. Craft the game prompt

A good game prompt has two parts: a system message that sets constraints, and a user message that describes the game.

The system message establishes output format and constraints:

```python
system_prompt = (
    "You are an expert game developer. You produce complete, self-contained "
    "HTML files with embedded CSS and JavaScript. Never use external CDNs, "
    "libraries, or dependencies. Use HTML5 Canvas for rendering. The game "
    "must be fully playable in any modern browser by opening the HTML file directly."
)
```

The user message describes the specific game. Be concrete about mechanics, controls, and visuals:

```python
user_prompt = (
    "Create a complete, playable game: A top-down dungeon crawler. "
    "The player navigates procedurally generated rooms connected by doorways. "
    "Each room contains enemies that patrol and chase the player on sight. "
    "Defeating enemies drops health potions or score pickups. The player has "
    "a melee attack (spacebar) and 3 lives. Generate at least 5 connected rooms. "
    "Show a minimap in the corner.\n\n"
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
```

The more specific the prompt, the better the result. Include details about movement, combat, scoring, and visual style.

## 3. Call the model and extract the code

Initialize the client and send the prompt to GLM. Code generation can take a few minutes, so set a generous timeout:

```python
import os
import re
from mistralai.client import Mistral
from dotenv import load_dotenv

load_dotenv()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

response = client.chat.complete(
    model="zai-glm-5-2",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    timeout_ms=600_000,
)
```

The `timeout_ms=600_000` parameter sets a 10-minute timeout. GLM generates large code outputs, so this headroom prevents the request from timing out.

Extract the HTML from the response. The model wraps its output in a ` ```html ` code fence:

```python
text = response.choices[0].message.content

match = re.search(r"```html\s*\n(.*?)```", text, re.DOTALL)
if match:
    html_content = match.group(1).strip()
else:
    # Fall back to DOCTYPE extraction
    match = re.search(r"(<!DOCTYPE.*?</html>)", text, re.DOTALL | re.IGNORECASE)
    html_content = match.group(1).strip()
```

## 4. Save and serve the game

Write the extracted HTML to a file, start a local server, and open it in your browser:

```python
import functools
import http.server
import webbrowser
from pathlib import Path

output = Path("game.html")
output.write_text(html_content, encoding="utf-8")

# Serve on localhost to avoid file:// security restrictions
handler = functools.partial(
    http.server.SimpleHTTPRequestHandler, directory=str(output.resolve().parent)
)
server = http.server.HTTPServer(("localhost", 8000), handler)

webbrowser.open(f"http://localhost:8000/{output.name}")
server.serve_forever()  # Ctrl+C to stop
```

Or run the included script directly:

```bash
python generate_game.py
```

The script calls GLM, extracts the HTML, saves it to `game.html`, starts a local server at `http://localhost:8000/game.html`, and opens it in your browser. Press Ctrl+C to stop the server when you're done playing.

## 5. Try different games

Change the game description in `generate_game.py` or call the function with a different prompt. Here are a few ideas:

**Space shooter:**

```python
from generate_game import generate_game

generate_game(
    "A vertical-scrolling space shooter. The player controls a ship at the bottom "
    "of the screen, moves left/right with arrow keys, and shoots with spacebar. "
    "Waves of enemy ships descend from the top with different movement patterns. "
    "Power-ups drop from destroyed enemies: rapid fire, shield, triple shot. "
    "Track score and display a high-score counter."
)
```

**Breakout clone:**

```python
generate_game(
    "A Breakout/Arkanoid clone. The player controls a paddle at the bottom with "
    "left/right arrow keys. A ball bounces around the screen destroying colored "
    "bricks. Different brick colors take different numbers of hits. Some bricks "
    "drop power-ups: wider paddle, multi-ball, sticky paddle. Include 3 levels "
    "with different brick layouts."
)
```

**Tower defense:**

```python
generate_game(
    "A tower defense game. Enemies follow a winding path from the top-left to "
    "the bottom-right. Click on empty tiles adjacent to the path to place towers. "
    "Three tower types: arrow (fast, low damage), cannon (slow, splash damage), "
    "and ice (slows enemies). Earn gold from defeated enemies to buy more towers. "
    "Survive 10 waves with increasing difficulty."
)
```

## Summary

You built a game generator that turns a text description into a playable browser game.

- Sent a structured prompt to GLM (`zai-glm-5-2`) requesting a self-contained HTML game
- Extracted the HTML from the model response using regex parsing
- Served the output on localhost and opened it in a browser

**Mistral features used:**

- Chat completions API with the `zai-glm-5-2` model
- System and user message roles for structured prompting
- Extended timeout (`timeout_ms`) for long code generation

Try describing your own game idea and see what GLM produces. For more on available models, see the [models documentation](https://docs.mistral.ai/getting-started/models/models_overview/).
