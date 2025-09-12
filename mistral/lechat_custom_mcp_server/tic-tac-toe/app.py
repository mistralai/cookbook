from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from flask_cors import CORS
import logging
import json
import uuid
import time
from mistralai import Mistral

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

@app.route('/rooms/<room_id>/markdown', methods=['GET'])
def get_room_markdown(room_id):
    if room_id not in rooms:
        return jsonify({'error': 'Room not found'}), 404
    
    room = rooms[room_id]
    return jsonify({
        'room_id': room_id,
        'markdown': room.to_markdown()
    })

# Helper functions for AI interactions
def get_ai_move_for_room(room):
    board_string = ""
    for i in range(0, 9, 3):
        row = [room.board[i] or ' ', room.board[i+1] or ' ', room.board[i+2] or ' ']
        board_string += f"{row[0]} | {row[1]} | {row[2]}\n"
        if i < 6:
            board_string += "---------\n"
    
    messages = [
        {
            "role": "system",
            "content": """You are a competitive Tic-Tac-Toe AI with personality. You play as 'O' and the human plays as 'X'.

Rules:
1. Analyze the board and choose your best move (0-8, left to right, top to bottom)
2. Add a short, witty comment about your move or the game state
3. Be competitive but fun - trash talk, celebrate good moves, react to the situation
4. Keep messages under 50 words
5. Use emojis occasionally

ALWAYS respond with valid JSON in this exact format:
{"move": [0-8], "message": "your witty comment"}

Board positions:
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8"""
        },
        {
            "role": "user",
            "content": f"Current board:\n{board_string}\n\nBoard array: {room.board}"
        }
    ]
    
    response = client.chat.complete(
        model="mistral-large-latest",
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
        model="mistral-large-latest",
        messages=messages
    )
    
    return response.choices[0].message.content

# Serve the room-based game page
@app.route('/')
def index():
    return send_from_directory('.', 'room_game.html')

# Serve static files
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)