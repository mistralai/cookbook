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

## How it works

The script generates games in three phases:

1. **Generate**: Send a detailed game prompt to GLM with engineering requirements that prevent common failure modes (broken collision, unkillable enemies, invalid spawning).
2. **Review and fix**: Send the generated HTML to `mistral-medium-latest` for a structured QA check. If issues are found, send them back to GLM for a targeted fix. Repeat up to 2 times.
3. **Serve**: Save the HTML and open it in your browser via a local HTTP server.

## Run

Generate a new game:

```bash
python generate_game.py
```

The script calls GLM, reviews the game for broken mechanics, fixes any issues (up to 2 rounds), saves the HTML to `game.html`, and opens it in your browser. Press Ctrl+C to stop the server.

Example output:

```
Generating game: A top-down dungeon crawler. The player navigates procedurally generated rooms...
This may take a few minutes...
Reviewing game (attempt 1/2)...
Issues found:
3. Combat: The attack function does not check collision against enemies.
4. Spawning: Enemies are placed at random positions without checking for wall overlap.
Fixing issues...
Reviewing game (attempt 2/2)...
Review passed.
Game saved to /Users/you/glm_game_generator/game.html
Serving game at http://localhost:8000/game.html
Press Ctrl+C to stop the server.
```

## Edit an existing game

If the generated game has a specific problem, use `--edit` to fix it without regenerating from scratch:

```bash
python generate_game.py --edit "enemies don't take damage when I attack them"
```

This reads the existing `game.html`, sends it to GLM with your feedback, and overwrites the file with the fix.

More examples:

```bash
python generate_game.py --edit "the minimap doesn't update when I move to a new room"
python generate_game.py --edit "add a boss enemy in the final room"
python generate_game.py --edit "make the player move faster and add a dash ability on shift"
```

## Try different games

Change `GAME_DESCRIPTION` in `generate_game.py` to generate a different game. See the [cookbook](01-glm-game-generator.md#try-different-games) for example descriptions (space shooter, breakout clone, tower defense).

## Summary

- Sent a structured prompt to GLM (`zai-glm-5-2`) requesting a self-contained HTML game
- Reviewed the output with `mistral-medium-latest` and auto-fixed broken mechanics
- Served the output on localhost and opened it in a browser

**Mistral features used:**

- Chat completions API with the `zai-glm-5-2` model for code generation
- Chat completions API with `mistral-medium-latest` for code review
- System and user message roles for structured prompting
- Extended timeout (`timeout_ms` and `httpx.Timeout`) for long code generation

Try describing your own game idea and see what GLM produces. For more on available models, see the [models documentation](https://docs.mistral.ai/getting-started/models/models_overview/).
