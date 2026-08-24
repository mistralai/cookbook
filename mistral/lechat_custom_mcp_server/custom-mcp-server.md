# Build a custom MCP server tic-tac-toe game for Vibe

Build a tic-tac-toe MCP server powered by Mistral, deploy it to Hugging Face Spaces, and connect it to [Vibe](https://chat.mistral.ai) to play directly from the chat interface.

## What you will build

The following table outlines the five steps in this cookbook:

| Step | What you do | Result |
|------|------------|--------|
| 1 | Build game logic and a Flask API | A REST backend that manages rooms, moves, and AI chat |
| 2 | Add an MCP server layer | Six tools Vibe can call over SSE |
| 3 | Containerize with Docker | A portable image ready for any host |
| 4 | Deploy to Hugging Face Spaces | A public SSE endpoint |
| 5 | Connect to Vibe | A working connector you can play from chat |

## How it works

The architecture follows a chain from Vibe through your MCP server to the game logic:

```text
Vibe  <-->  SSE Transport  <-->  Your MCP Server  <-->  Game Logic + Mistral
```

1. You type "create a new tic-tac-toe game" in Vibe
2. Vibe calls your MCP server's `create_room()` tool
3. Your server creates a game session and returns the board
4. You make a move — Vibe calls `make_move(position)`
5. Your server processes the move, asks Mistral for a counter-move
6. The updated board and some AI trash talk come back
7. Play continues until someone wins or it's a draw

## Prerequisites

To complete this cookbook, you will need:

- Python 3.11+
- A Mistral account and API key
- A [Hugging Face](https://huggingface.co) account with a Pro subscription (to deploy the Docker container)

## Environment setup

### Create a Hugging Face Space

Create the Space first so you can build your project files directly inside the cloned repository.

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Name your Space (for example, `tictactoe-mcp-server`)
3. Select **Docker** as the SDK
4. Set visibility to **Public** (required for Vibe connectors)

### Required environment variables

You'll need a Mistral API key. In [Studio](https://console.mistral.ai), navigate to the [API keys section](https://console.mistral.ai/home?profile_dialog=api-keys), select **Private and shared connectors** from the **Connectors access scope** dropdown menu, and create a new key.

Add your API key as a Space secret so it's available at runtime without being stored in the repository:

1. Go to your Space's **Settings** tab
2. Scroll to **Variables and secrets**
3. Click **New secret**
4. Set the name to `MISTRAL_API_KEY` and paste your API key as the value

Space secrets are write-only — once saved, the value can't be read from the settings page. They're injected as environment variables at runtime, so `os.getenv('MISTRAL_API_KEY')` in your code works the same way it does with a local `.env` file. Don't commit a `.env` file to your repository.

### Clone the Space

To clone and push to Hugging Face, you need a [User Access Token](https://huggingface.co/docs/hub/security-tokens#user-access-tokens) with **write** permissions. Generate one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Git will prompt for your credentials when you clone — use your Hugging Face username and the access token as your password.

Clone the Space repository to your machine. Replace `<your-username>` with your Hugging Face username and `<space-name>` with the name you chose:

```bash
git clone https://huggingface.co/spaces/<your-username>/<space-name>
cd <space-name>
```

### Install

Create a `requirements.txt` and install the dependencies in one step:

```bash
echo "fastmcp\nflask\nflask-cors\nmistralai\npython-dotenv" > requirements.txt && pip install -r requirements.txt
```

All files in the following steps are created inside this directory.

---

## Step 1 — Game logic and Flask API

Create `app.py`. This file handles all game state and exposes four REST endpoints that the MCP layer calls internally.

```python
from flask import Flask, request, jsonify
import os
from flask_cors import CORS
import logging
import json
import uuid
import time
from mistralai.client import Mistral

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Mistral client
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
if not MISTRAL_API_KEY:
    logger.error("MISTRAL_API_KEY not configured")
    exit(1)

client = Mistral(api_key=MISTRAL_API_KEY)

class Room:
    def __init__(self, room_id=None):
        self.id = room_id or str(uuid.uuid4())[:8]
        self.board = [''] * 9
        self.current_player = 'X'  # X = human, O = AI
        self.game_status = 'active'  # 'active', 'won', 'draw'
        self.winner = None
        self.chat_history = []
        self.created = time.time()
        self.last_activity = time.time()
        self.moves_count = 0

        # Add welcome message
        self.chat_history.append({
            'sender': 'ai',
            'message': "Hey there! Ready for a game of Tic-Tac-Toe? I'm pretty good at this... 😏 You're X, I'm O. Good luck!",
            'timestamp': time.time()
        })

    def make_move(self, position, player):
        if self.game_status != 'active' or self.board[position] != '':
            return False

        self.board[position] = player
        self.moves_count += 1
        self.last_activity = time.time()

        # Check for winner
        if self.check_winner():
            self.game_status = 'won'
            self.winner = player
        elif self.moves_count == 9:
            self.game_status = 'draw'
        else:
            self.current_player = 'O' if player == 'X' else 'X'

        return True

    def check_winner(self):
        win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]

        for pattern in win_patterns:
            a, b, c = pattern
            if self.board[a] and self.board[a] == self.board[b] == self.board[c]:
                return True
        return False

    def add_chat_message(self, message, sender):
        self.chat_history.append({
            'sender': sender,
            'message': message,
            'timestamp': time.time()
        })
        self.last_activity = time.time()

    def to_markdown(self):
        # Game header
        markdown = f"# Game Room: {self.id}\n"
        markdown += f"## Status: "

        if self.game_status == 'won':
            winner_name = "You" if self.winner == 'X' else "Mistral AI"
            markdown += f"Game Over - {winner_name} wins! 🎉\n"
        elif self.game_status == 'draw':
            markdown += "Game Over - It's a draw! 🤝\n"
        else:
            turn_name = "Your turn" if self.current_player == 'X' else "Mistral's turn"
            markdown += f"{turn_name} ({self.current_player} to play)\n"

        markdown += f"Moves: {self.moves_count}/9\n\n"

        # Board representation
        markdown += "```\n"
        for i in range(0, 9, 3):
            row = [self.board[i] or '·', self.board[i+1] or '·', self.board[i+2] or '·']
            markdown += f"{row[0]} | {row[1]} | {row[2]}\n"
            if i < 6:
                markdown += "-----------\n"
        markdown += "```\n\n"

        # Chat history (last 5 messages)
        if self.chat_history:
            markdown += "## Recent Chat\n"
            recent_messages = self.chat_history[-5:]
            for msg in recent_messages:
                sender_name = "**You:**" if msg['sender'] == 'user' else "**Mistral AI:**"
                markdown += f"{sender_name} {msg['message']}\n"

        return markdown

    def to_dict(self):
        return {
            'id': self.id,
            'board': self.board,
            'current_player': self.current_player,
            'game_status': self.game_status,
            'winner': self.winner,
            'chat_history': self.chat_history,
            'moves_count': self.moves_count,
            'created': self.created,
            'last_activity': self.last_activity
        }

# In-memory room storage
rooms = {}

# Room management endpoints
@app.route('/rooms', methods=['POST'])
def create_room():
    room = Room()
    rooms[room.id] = room
    logger.info(f"Created room: {room.id}")
    return jsonify({
        'room_id': room.id,
        'status': 'created',
        'room_data': room.to_dict()
    })

@app.route('/rooms/<room_id>', methods=['GET'])
def get_room(room_id):
    if room_id not in rooms:
        return jsonify({'error': 'Room not found'}), 404

    room = rooms[room_id]
    return jsonify({
        'room_id': room_id,
        'room_data': room.to_dict(),
        'markdown': room.to_markdown()
    })

@app.route('/rooms/<room_id>/move', methods=['POST'])
def make_room_move(room_id):
    if room_id not in rooms:
        return jsonify({'error': 'Room not found'}), 404

    room = rooms[room_id]
    data = request.json
    position = data.get('position')

    if position is None or position < 0 or position > 8:
        return jsonify({'error': 'Invalid position'}), 400

    # Make human move
    if not room.make_move(position, 'X'):
        return jsonify({'error': 'Invalid move'}), 400

    # Check if game ended
    if room.game_status != 'active':
        return jsonify({
            'room_data': room.to_dict(),
            'markdown': room.to_markdown(),
            'ai_move': None
        })

    # Get AI move
    try:
        ai_response = get_ai_move_for_room(room)
        if ai_response and 'move' in ai_response:
            # Validate AI move
            ai_move = ai_response['move']
            if 0 <= ai_move <= 8 and room.board[ai_move] == '':
                room.make_move(ai_move, 'O')
                if 'message' in ai_response:
                    room.add_chat_message(ai_response['message'], 'ai')
            else:
                logger.error(f"AI chose invalid move: {ai_move}, board: {room.board}")
                # Fallback to random valid move
                empty_positions = [i for i in range(9) if room.board[i] == '']
                if empty_positions:
                    fallback_move = empty_positions[0]  # Take first available
                    room.make_move(fallback_move, 'O')
                    room.add_chat_message("Oops, had a brain freeze! But I'm still playing! 🤖", 'ai')

        return jsonify({
            'room_data': room.to_dict(),
            'markdown': room.to_markdown(),
            'ai_move': ai_response
        })
    except Exception as e:
        logger.error(f"AI move failed: {e}")
        # Fallback to random valid move instead of failing
        empty_positions = [i for i in range(9) if room.board[i] == '']
        if empty_positions:
            fallback_move = empty_positions[0]
            room.make_move(fallback_move, 'O')
            room.add_chat_message("Technical difficulties, but I'm improvising! 😅", 'ai')

        return jsonify({
            'room_data': room.to_dict(),
            'markdown': room.to_markdown(),
            'ai_move': {'move': fallback_move if empty_positions else None, 'message': 'Technical difficulties!'}
        })

@app.route('/rooms/<room_id>/chat', methods=['POST'])
def room_chat(room_id):
    if room_id not in rooms:
        return jsonify({'error': 'Room not found'}), 404

    room = rooms[room_id]
    data = request.json
    user_message = data.get('message', '')

    if not user_message.strip():
        return jsonify({'error': 'Empty message'}), 400

    # Add user message
    room.add_chat_message(user_message, 'user')

    # Get AI response
    try:
        ai_response = get_ai_chat_for_room(room, user_message)
        room.add_chat_message(ai_response, 'ai')

        return jsonify({
            'room_data': room.to_dict(),
            'markdown': room.to_markdown(),
            'ai_response': ai_response
        })
    except Exception as e:
        logger.error(f"AI chat failed: {e}")
        return jsonify({'error': 'AI chat failed'}), 500

# Helper functions for AI interactions
def get_ai_move_for_room(room):
    board_string = ""
    for i in range(0, 9, 3):
        row = [room.board[i] or ' ', room.board[i+1] or ' ', room.board[i+2] or ' ']
        board_string += f"{row[0]} | {row[1]} | {row[2]}\n"
        if i < 6:
            board_string += "---------\n"

    empty_positions = [i for i in range(9) if room.board[i] == '']

    messages = [
        {
            "role": "system",
            "content": f"""You are a competitive Tic-Tac-Toe AI with personality. You play as 'O' and the human plays as 'X'.

Rules:
1. You MUST choose from these available positions ONLY: {empty_positions}
2. Add a short, witty comment about your move or the game state
3. Be competitive but fun - trash talk, celebrate good moves, react to the situation
4. Keep messages under 50 words
5. Use emojis occasionally

ALWAYS respond with valid JSON in this exact format:
{{"move": <one of {empty_positions}>, "message": "your witty comment"}}

Board positions:
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8"""
        },
        {
            "role": "user",
            "content": f"Current board:\n{board_string}\n\nAvailable positions: {empty_positions}\n\nBoard array: {room.board}"
        }
    ]

    response = client.chat.complete(
        model="mistral-medium-latest",
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

def get_ai_chat_for_room(room, user_message):
    board_string = ""
    for i in range(0, 9, 3):
        row = [room.board[i] or ' ', room.board[i+1] or ' ', room.board[i+2] or ' ']
        board_string += f"{row[0]} | {row[1]} | {row[2]}\n"
        if i < 6:
            board_string += "---------\n"

    messages = [
        {
            "role": "system",
            "content": f"""You are a competitive, witty Tic-Tac-Toe AI with personality. You're currently playing a game.

Current board state:
{board_string}

Respond to the human's message with personality - be competitive, funny, encouraging, or trash-talking as appropriate.
Keep responses under 50 words. Use emojis occasionally. Don't make game moves in chat - that happens separately."""
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    response = client.chat.complete(
        model="mistral-medium-latest",
        messages=messages
    )

    return response.choices[0].message.content

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
```

The `Room` class tracks board state, move history, and chat messages. Each room gets a short UUID and supports the full game lifecycle: creating, making moves, checking winners, and chatting with the AI opponent.

The two AI helper functions use `mistral-medium-latest` with different system prompts. `get_ai_move_for_room` uses JSON mode to get a structured move and trash-talk message — the prompt explicitly lists which board positions are still empty so the model doesn't pick an occupied square. `get_ai_chat_for_room` handles freeform conversation during the game.

The four Flask endpoints expose this logic as a REST API:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/rooms` | POST | Create a new game room |
| `/rooms/<room_id>` | GET | Get room state and board |
| `/rooms/<room_id>/move` | POST | Make a move (triggers AI counter-move) |
| `/rooms/<room_id>/chat` | POST | Chat with the AI opponent |

## Step 2 — MCP server layer

Create `mcp_server.py`. This wraps the game logic in MCP tools that Vibe can discover and call over SSE.

```python
import os
import asyncio
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from app import Room, rooms, get_ai_move_for_room, get_ai_chat_for_room

load_dotenv()

# --- MCP Server Setup ---
mcp = FastMCP(
    name="TicTacToeRooms",
    host="0.0.0.0",
    port=7860,
)

# --- Global state for current user session ---
current_session = {
    'active_room_id': None,
    'username': 'MCPPlayer'
}

# --- MCP Tools ---

@mcp.tool()
def create_room() -> dict:
    """
    Create a new tic-tac-toe game room.
    Returns:
        dict: Room information including room ID and initial markdown state
    """
    global current_session
    try:
        room = Room()
        rooms[room.id] = room

        current_session['active_room_id'] = room.id

        return {
            "status": "success",
            "room_id": room.id,
            "message": f"Created new tic-tac-toe room: {room.id}",
            "markdown_state": room.to_markdown(),
            "instructions": "Use make_move() to play or send_chat() to talk with Mistral AI",
            "game_info": {
                "your_symbol": "X",
                "ai_symbol": "O",
                "board_positions": "0-8 (left to right, top to bottom)"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create room: {str(e)}"
        }

@mcp.tool()
def get_room_state(room_id: str = None) -> dict:
    """
    Get the current state of a tic-tac-toe room in markdown format.
    Args:
        room_id (str, optional): Room ID to check (uses current active room if not provided)
    Returns:
        dict: Current room state with markdown representation
    """
    global current_session
    try:
        # Use provided room_id or current active room
        target_room_id = room_id or current_session.get('active_room_id')

        if not target_room_id:
            return {
                "status": "error",
                "message": "No active room. Create a room first using create_room()."
            }

        if target_room_id not in rooms:
            return {
                "status": "error",
                "message": f"Room {target_room_id} not found. It may have been cleaned up."
            }

        room = rooms[target_room_id]

        return {
            "status": "success",
            "room_id": target_room_id,
            "markdown_state": room.to_markdown(),
            "game_status": room.game_status,
            "current_player": room.current_player,
            "moves_made": room.moves_count,
            "your_turn": room.current_player == 'X' and room.game_status == 'active'
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get room state: {str(e)}"
        }

@mcp.tool()
async def make_move(position: int, room_id: str = None) -> dict:
    """
    Make a move in a tic-tac-toe game. This will also trigger the AI's response move.
    Args:
        position (int): Board position (0-8, left to right, top to bottom)
        room_id (str, optional): Room ID (uses current active room if not provided)
    Returns:
        dict: Result of your move and the AI's response with updated game state
    """
    global current_session
    try:
        # Use provided room_id or current active room
        target_room_id = room_id or current_session.get('active_room_id')

        if not target_room_id:
            return {
                "status": "error",
                "message": "No active room. Create a room first using create_room()."
            }

        if target_room_id not in rooms:
            return {
                "status": "error",
                "message": f"Room {target_room_id} not found."
            }

        room = rooms[target_room_id]

        # Validate move
        if position < 0 or position > 8:
            return {
                "status": "error",
                "message": "Invalid position. Use 0-8 (left to right, top to bottom)."
            }

        if room.game_status != 'active':
            return {
                "status": "error",
                "message": f"Game is over. Status: {room.game_status}",
                "markdown_state": room.to_markdown()
            }

        if room.current_player != 'X':
            return {
                "status": "error",
                "message": "It's not your turn! Wait for AI to move.",
                "markdown_state": room.to_markdown()
            }

        # Make human move
        if not room.make_move(position, 'X'):
            return {
                "status": "error",
                "message": f"Invalid move! Position {position} may already be occupied.",
                "markdown_state": room.to_markdown()
            }

        result_message = f"✅ You played X at position {position}\n\n"

        # Check if game ended after human move
        if room.game_status != 'active':
            if room.winner == 'X':
                result_message += "🎉 Congratulations! You won!\n\n"
            else:
                result_message += "🤝 It's a draw!\n\n"

            result_message += room.to_markdown()
            return {
                "status": "success",
                "message": result_message,
                "game_over": True,
                "winner": room.winner
            }

        # Get AI move
        try:
            ai_response = get_ai_move_for_room(room)
            if ai_response and 'move' in ai_response:
                # Validate AI move
                ai_move = ai_response['move']
                if 0 <= ai_move <= 8 and room.board[ai_move] == '':
                    room.make_move(ai_move, 'O')
                    if 'message' in ai_response:
                        room.add_chat_message(ai_response['message'], 'ai')
                else:
                    # Fallback to first available position
                    empty_positions = [i for i in range(9) if room.board[i] == '']
                    if empty_positions:
                        ai_move = empty_positions[0]
                        ai_response['move'] = ai_move
                        room.make_move(ai_move, 'O')
                        room.add_chat_message("Oops, had a brain freeze! But I'm still playing!", 'ai')

                result_message += f"🤖 Mistral AI played O at position {ai_response['move']}\n"
                if 'message' in ai_response:
                    result_message += f"💬 Mistral says: \"{ai_response['message']}\"\n\n"
                else:
                    result_message += "\n"

                # Check if AI won
                if room.game_status == 'won' and room.winner == 'O':
                    result_message += "💀 Mistral AI wins this round!\n\n"
                elif room.game_status == 'draw':
                    result_message += "🤝 It's a draw!\n\n"
            else:
                result_message += "⚠️ AI move failed, but you can continue\n\n"

        except Exception as e:
            result_message += f"⚠️ AI move error: {str(e)}\n\n"

        result_message += room.to_markdown()

        return {
            "status": "success",
            "message": result_message,
            "game_over": room.game_status != 'active',
            "winner": room.winner if room.game_status == 'won' else None,
            "your_turn": room.current_player == 'X' and room.game_status == 'active'
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to make move: {str(e)}"
        }

@mcp.tool()
async def send_chat(message: str, room_id: str = None) -> dict:
    """
    Send a chat message to Mistral AI in the current game room.
    Args:
        message (str): Your message to send to the AI
        room_id (str, optional): Room ID (uses current active room if not provided)
    Returns:
        dict: Your message and the AI's response with updated room state
    """
    global current_session
    try:
        # Use provided room_id or current active room
        target_room_id = room_id or current_session.get('active_room_id')

        if not target_room_id:
            return {
                "status": "error",
                "message": "No active room. Create a room first using create_room()."
            }

        if target_room_id not in rooms:
            return {
                "status": "error",
                "message": f"Room {target_room_id} not found."
            }

        room = rooms[target_room_id]

        # Add user message
        room.add_chat_message(message, 'user')

        # Get AI response
        ai_response = get_ai_chat_for_room(room, message)
        room.add_chat_message(ai_response, 'ai')

        result_message = f"💬 **You:** {message}\n💬 **Mistral AI:** {ai_response}\n\n"
        result_message += room.to_markdown()

        return {
            "status": "success",
            "message": result_message,
            "your_message": message,
            "ai_response": ai_response
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to send chat: {str(e)}"
        }

@mcp.tool()
def list_rooms() -> dict:
    """
    List all active tic-tac-toe game rooms.
    Returns:
        dict: List of active rooms with their status
    """
    try:
        if not rooms:
            return {
                "status": "success",
                "message": "No active rooms. Use create_room() to start a new game!",
                "active_rooms": [],
                "count": 0
            }

        room_list = []
        for room_id, room in rooms.items():
            room_info = {
                "room_id": room_id,
                "game_status": room.game_status,
                "current_player": room.current_player,
                "moves_count": room.moves_count,
                "winner": room.winner,
                "is_your_turn": room.current_player == 'X' and room.game_status == 'active',
                "is_active": current_session.get('active_room_id') == room_id
            }
            room_list.append(room_info)

        active_room_id = current_session.get('active_room_id')
        message = f"Found {len(room_list)} active rooms."
        if active_room_id:
            message += f" Current active room: {active_room_id}"

        return {
            "status": "success",
            "message": message,
            "active_rooms": room_list,
            "count": len(room_list),
            "current_active_room": active_room_id
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to list rooms: {str(e)}"
        }

@mcp.tool()
def get_help() -> dict:
    """
    Get help information about playing tic-tac-toe.
    Returns:
        dict: Instructions and tips for playing the game
    """
    return {
        "status": "success",
        "message": "Tic-Tac-Toe Game Help",
        "instructions": {
            "how_to_play": [
                "1. Create a new game room with create_room()",
                "2. Make moves using make_move(position) where position is 0-8",
                "3. Chat with Mistral AI using send_chat('your message')",
                "4. Check game state anytime with get_room_state()"
            ],
            "board_layout": {
                "description": "Board positions (0-8):",
                "layout": [
                    "0 | 1 | 2",
                    "---------",
                    "3 | 4 | 5",
                    "---------",
                    "6 | 7 | 8"
                ]
            },
            "symbols": {
                "you": "X (you go first)",
                "ai": "O (Mistral AI)"
            },
            "tips": [
                "The AI has personality and will trash talk!",
                "You can have multiple rooms active at once",
                "Use list_rooms() to see all your games"
            ]
        },
        "available_commands": [
            "create_room() - Start a new game",
            "make_move(position) - Make your move (0-8)",
            "send_chat('message') - Chat with AI",
            "get_room_state() - Check current game",
            "list_rooms() - See all active games",
            "get_help() - Show this help"
        ]
    }

# --- Server Execution ---
if __name__ == "__main__":
    print(f"Tic-Tac-Toe Rooms MCP Server starting on port 7860...")
    print("Available game features:")
    print("- Create multiple game rooms")
    print("- Play against Mistral AI with personality")
    print("- Real-time chat with the AI")
    print("- Markdown state representation")
    print("- Room management")
    print()
    print("MCP Tools available:")
    print("- create_room()")
    print("- make_move(position)")
    print("- send_chat(message)")
    print("- get_room_state()")
    print("- list_rooms()")
    print("- get_help()")
    print()
    print("This MCP server is ready for Vibe integration!")
    print("Running Tic-Tac-Toe MCP server with SSE transport")
    mcp.run(transport="sse")
```

FastMCP handles SSE transport, tool discovery, and schema generation. The `@mcp.tool()` decorator exposes each function as a callable tool. Vibe discovers these tools automatically when you register the connector.

The six tools and what they do:

| Tool | Purpose |
|------|---------|
| `create_room()` | Start a new game session |
| `get_room_state()` | Check the current board and status |
| `make_move(position)` | Play your move (triggers AI counter-move) |
| `send_chat(message)` | Chat with the AI during the game |
| `list_rooms()` | See all active games |
| `get_help()` | Get instructions and board layout |

## Step 3 — Containerize with Docker

Create a `Dockerfile` to package everything for deployment.

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Create startup script that runs MCP server only (simpler approach)
RUN echo '#!/bin/bash\n\
echo "Starting Tic-Tac-Toe MCP Server on port 7860..."\n\
echo "This is the MCP server for Vibe integration"\n\
echo "MCP Tools: create_room, make_move, send_chat, get_room_state, list_rooms"\n\
python mcp_server.py' > start.sh && chmod +x start.sh

CMD ["./start.sh"]
```

The container runs `mcp_server.py` (which imports from `app.py`), exposing the SSE endpoint on port 7860 — the default port for Hugging Face Spaces.

## Step 4 — Deploy to Hugging Face Spaces

Push your files to the Space repository you cloned in Step 1:

```bash
git add app.py mcp_server.py requirements.txt Dockerfile
git commit -m "Add tic-tac-toe MCP server"
git push
```

Monitor the build in the **Logs** tab of your Space. Once the status shows **Running**, your SSE endpoint is live at:

```text
https://<your-username>-<space-name>.hf.space/sse
```

## Step 5 — Connect to Vibe

You can register your MCP server in [Studio](https://console.mistral.ai/build/connectors) or programmatically.

### Registering the MCP server in Studio

To register the server in Studio: 

1. Navigate to the [Connectors](https://console.mistral.ai/build/connectors) tab and click **Add Connector**.
2. Click the **Custom MCP Server** tab on the modal.
3. Give your connector a title like `Tic Tac Toe`.
4. Add your space's URL in the proper format: `https://<your-username>-<space-name>.hf.space/sse`
5. Optionally provide a description like "Play tic-tac-toe against Mistral".

### Registering the MCP server programmatically using the Mistral API

Register your deployed MCP server as a Vibe connector using the Mistral API.

```python
from mistralai.client import Mistral
import os

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

connector = client.beta.connectors.create(
    name="tictactoe-mcp",
    url="https://<your-username>-<space-name>.hf.space/sse",
    description="Play tic-tac-toe against Mistral",
    visibility="private",
)

print(f"Connector ID: {connector.id}")
print(f"Connector name: {connector.name}")

# This MCP server doesn't require authentication, 
# but you must add credentials to enable the server on your project.
await client.beta.connectors.create_or_update_user_credentials_async(
    connector_id_or_name="connector.name",
    name=f"new-credential",
    credentials={'headers':{}},
    is_default=True,
)
```

Replace the URL with your actual Hugging Face Spaces endpoint.

#### Verify the Connector

Once authenticated, verify that Vibe can reach your server:

```python
connector_info = client.beta.connectors.get(connector_id=connector.id)
print(f"Status: {connector_info.status}")
print(f"Tools: {connector_info.tools}")
```

You should see all six tools listed.

## Play in Vibe

1. Open [Vibe](https://chat.mistral.ai)
2. In a new conversation, enable your `tictactoe-mcp` connector
3. Type "Let's play tic-tac-toe" and watch Vibe call your MCP tools

## Clean up

To avoid unnecessary resource usage, delete the connector and the Hugging Face Space when you're done.

### Delete the connector

Remove the connector in [Studio](https://console.mistral.ai/build/connectors) by clicking the Connector, selecting the three dots, and selecting **Delete**, or programmatically:

```python
client.beta.connectors.delete(connector_id=connector.id)
```

### Delete the Hugging Face Space

1. Go to your Space's **Settings** tab
2. Scroll to the bottom and click **Delete this Space**

## Summary

This cookbook walked through building a tic-tac-toe MCP server from scratch and connecting it to Vibe as a playable connector.

**What you built:**

- A Flask REST API with game room management and AI-powered moves
- An MCP server layer exposing six tools over SSE with FastMCP
- A Docker container deployed to Hugging Face Spaces
- A Vibe connector registered through the Connectors API

**Mistral features used:**

- Chat Completions API with JSON mode for structured AI moves
- Chat Completions API for freeform in-game conversation
- Connectors API for registering the MCP server with Vibe

View your Connectors in [Studio](https://console.mistral.ai/build/connectors).
