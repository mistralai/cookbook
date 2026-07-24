from dotenv import load_dotenv
from .configs import goals_set, food_logs, mistral_model, client

load_dotenv()

DAILY_PROGRESS_TOOL = {
            "type": "function",
            "function": {
                "name": "daily_progress",
                "description": "Provide daily progress for the food taken by user so far.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "user_query",
                    ]
                },
            },
        }

def daily_progress(user_query: str):
    """Provide daily progress of the user based on his goals and food intake."""

    global goals_set, food_logs
    prompt = f"""
    You are a nutrition analysis assistant. Provide an insightful analysis of the user's food logs.

    User goals: {goals_set}
    Today's logs: {food_logs}

    Analyze the following:
    1. Total calories and macronutrients consumed today.
    2. Progress towards daily goals (percentage).
    3. Identify any nutritional strengths or gaps.
    4. Provide helpful insights based on their eating patterns.
    5. Provide answer in short and concise manner.

    If they haven't logged any food today, suggest they log some meals first.
    If they haven't set goals, encourage them to set goals for more meaningful analysis.

    User Query: {user_query}
    """

    response = client.chat.complete(
            model=mistral_model,
            messages=[{"role": "user", "content": prompt}]
        )
    full_response = response.choices[0].message.content

    return full_response