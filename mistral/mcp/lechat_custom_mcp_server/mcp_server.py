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
    print("This MCP server is ready for LeChat integration!")
    print("Running Tic-Tac-Toe MCP server with SSE transport")
    mcp.run(transport="sse")