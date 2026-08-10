"""Generate a playable HTML5 mini game using the GLM model via the Mistral API."""

import argparse
import functools
import http.server
import os
import re
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

import httpx

from mistralai.client import Mistral

load_dotenv()

# Step 1 — Initialize the client
# GLM generates large code outputs that can take several minutes. The default
# httpx timeout is too short, so set it to 10 minutes to match timeout_ms.
client = Mistral(
    api_key=os.environ["MISTRAL_API_KEY"],
    timeout_ms=600_000,
    client=httpx.Client(follow_redirects=True, timeout=httpx.Timeout(600.0)),
)

# Step 2 — Craft the game prompt
# Be specific about mechanics, controls, visuals, and scope.
# The more detail you provide, the better the generated game.
GAME_DESCRIPTION = (
    "A top-down dungeon crawler. The player navigates procedurally generated "
    "rooms connected by doorways. Each room contains enemies that patrol and "
    "chase the player on sight. Defeating enemies drops health potions or score "
    "pickups. The player has a melee attack (spacebar) and 3 lives. Generate at "
    "least 5 connected rooms. Show a minimap in the corner."
)


# The system message constrains the output format (single HTML file, no
# external dependencies, Canvas rendering). The user message describes the
# game and lists concrete requirements so the model doesn't omit features.
# The "Game engineering requirements" block addresses common failure modes
# like broken collision detection, enemies that can't be killed, and missing
# spawn logic.
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
        "Game engineering requirements (follow these exactly):\n"
        "- Collision detection: implement rectangle or circle collision checks. "
        "Every entity (player, enemies, projectiles, items) must have x, y, "
        "width, and height properties used in collision tests.\n"
        "- Enemy health: every enemy must have a numeric health property that "
        "decreases when the player attacks. Remove the enemy when health "
        "reaches 0.\n"
        "- Combat feedback: when the player attacks, check collision against "
        "every enemy in range. On hit, decrease enemy health and show visual "
        "feedback (flash, particle, or color change).\n"
        "- Valid spawning: enemies must spawn on valid floor positions, never "
        "inside walls or on top of the player. Validate positions before "
        "placing.\n"
        "- Game loop integrity: the update function must call enemy AI, "
        "collision detection, and rendering every frame. Never skip a step.\n"
        "- Input handling: use keydown/keyup events with a keys-pressed "
        "object (e.g., `keys = {}`) that tracks which keys are currently held. "
        "Check this object each frame in the update loop.\n\n"
        "Return the complete HTML file inside a single ```html code fence."
    )
    return system_prompt, user_prompt


# Step 3 — Generate the game
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


# Wraps the GLM call and HTML extraction into a single function.
def generate_game(system_prompt: str, user_prompt: str) -> str:
    """Call GLM to generate a game and return the extracted HTML."""
    response = client.chat.complete(
        model="zai-glm-5-2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return extract_html(response.choices[0].message.content)


# Step 4 — Review and fix the game
# After generation, send the HTML to mistral-medium-latest for a structured
# review. If issues are found, send the HTML and issue list back to GLM for
# a targeted fix. This loop runs up to 2 times.
def review_game(html_content: str) -> str | None:
    """Review generated HTML for common game mechanic issues.

    Returns a string describing the issues found, or None if no issues.
    """
    review_prompt = (
        "You are a game QA engineer. Review the following HTML5 game for "
        "both runtime errors and broken game logic.\n\n"
        "PART 1 — RUNTIME ERRORS\n"
        "Trace these code paths from call site to implementation. Verify "
        "that every variable and property referenced actually exists. A "
        "function that accesses undefined properties is a FAIL.\n\n"
        "1. Initialization: Trace the startup path. Does every function "
        "called during init receive the arguments it expects? Are arrays "
        "and objects initialized before being accessed?\n"
        "2. Spawning: Trace the enemy spawn function. Does it access "
        "properties (like room.x, room.width) that actually exist on the "
        "objects passed to it?\n"
        "3. Game loop: Does the update/render loop call functions with "
        "correct arguments? Does it access properties on objects that "
        "might be undefined?\n"
        "4. Room transitions: When the player moves to a new room, are "
        "all references updated correctly?\n\n"
        "PART 2 — GAME LOGIC\n"
        "Trace these mechanics end-to-end. It's not enough for the code "
        "to exist — follow the logic and confirm it produces the correct "
        "outcome.\n\n"
        "5. Enemy death: Trace from player attack to enemy removal. Does "
        "the attack decrease enemy health? When health reaches 0, is the "
        "enemy actually removed from the array/list so it stops rendering "
        "and updating? A health property that decreases but never triggers "
        "removal is a FAIL.\n"
        "6. Collision detection: Are collision checks called with the "
        "correct coordinates and dimensions? Do entities have the x, y, "
        "width, height properties the checks reference?\n"
        "7. Combat feedback: When the player attacks and hits an enemy, "
        "is there any visual feedback (flash, color change, particle)? "
        "An attack that silently reduces health with no indication is a "
        "FAIL.\n"
        "8. Input handling: Are keydown/keyup events tracked in a "
        "keys-pressed object checked each frame? A system that only uses "
        "keydown without tracking held keys will miss continuous input.\n\n"
        "If ALL checks pass, respond with exactly: PASS\n\n"
        "If any check fails, describe the specific bug: which function, "
        "which property or logic path, and what goes wrong. Do not include "
        "the game code in your response.\n\n"
        f"```html\n{html_content}\n```"
    )
    response = client.chat.complete(
        model="mistral-medium-latest",
        messages=[{"role": "user", "content": review_prompt}],
    )
    result = response.choices[0].message.content.strip()
    if result.upper().startswith("PASS"):
        return None
    return result


def fix_game(html_content: str, issues: str) -> str:
    """Send the HTML and issue list back to GLM for a targeted fix."""
    fix_prompt = (
        "The following HTML5 game has specific issues that need fixing. "
        "Fix ONLY the listed issues. Keep everything else unchanged.\n\n"
        f"Issues to fix:\n{issues}\n\n"
        f"Game code:\n```html\n{html_content}\n```\n\n"
        "Return the complete fixed HTML file inside a single ```html code fence."
    )
    response = client.chat.complete(
        model="zai-glm-5-2",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert game developer. Fix the specific issues "
                    "listed in the game code. Return the complete, corrected "
                    "HTML file. Do not remove working features."
                ),
            },
            {"role": "user", "content": fix_prompt},
        ],
    )
    return extract_html(response.choices[0].message.content)


# Step 5 — Edit the game
# The --edit flag lets users describe what's wrong with an existing game
# and get a targeted fix without regenerating from scratch.
def edit_game(html_content: str, user_feedback: str) -> str:
    """Send existing game HTML and user feedback to GLM for a targeted fix."""
    edit_prompt = (
        "The following HTML5 game needs changes based on user feedback. "
        "Apply the requested changes while keeping everything else intact.\n\n"
        f"User feedback: {user_feedback}\n\n"
        f"Current game code:\n```html\n{html_content}\n```\n\n"
        "Return the complete updated HTML file inside a single ```html code fence."
    )
    response = client.chat.complete(
        model="zai-glm-5-2",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert game developer. Apply the user's "
                    "requested changes to the game code. Return the complete, "
                    "updated HTML file. Do not remove working features."
                ),
            },
            {"role": "user", "content": edit_prompt},
        ],
    )
    return extract_html(response.choices[0].message.content)


# Step 6 — Serve the game locally
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
    parser = argparse.ArgumentParser(description="Generate or edit an HTML5 game.")
    parser.add_argument(
        "--edit",
        type=str,
        help="Edit an existing game.html. Describe what to fix.",
    )
    args = parser.parse_args()

    output = Path("game.html")

    if args.edit:
        # Step 5 — Edit mode: read existing game and apply fixes
        if not output.exists():
            print(f"Error: {output} not found. Generate a game first.")
            return
        print(f"Editing game: {args.edit}")
        html_content = output.read_text(encoding="utf-8")
        html_content = edit_game(html_content, args.edit)
    else:
        # Step 2 — Craft the prompt
        system_prompt, user_prompt = build_game_prompt(GAME_DESCRIPTION)

        # Step 3 — Generate the game
        print(f"Generating game: {GAME_DESCRIPTION}")
        print("This may take a few minutes...")
        html_content = generate_game(system_prompt, user_prompt)

        # Step 4 — Review and fix
        for attempt in range(2):
            print(f"Reviewing game (attempt {attempt + 1}/2)...")
            issues = review_game(html_content)
            if issues is None:
                print("Review passed.")
                break
            print(f"Issues found:\n{issues}")
            print("Fixing issues...")
            html_content = fix_game(html_content, issues)
        else:
            print("Applied 2 rounds of fixes. Saving result.")

    output.write_text(html_content, encoding="utf-8")
    print(f"Game saved to {output.resolve()}")

    # Step 6 — Serve the game locally
    serve_and_open(output.resolve().parent, output.name)


if __name__ == "__main__":
    main()
