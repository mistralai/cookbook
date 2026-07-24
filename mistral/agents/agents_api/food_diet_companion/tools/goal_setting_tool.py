from typing import Dict, List
from pydantic import BaseModel, Field
import json
from .configs import goals_set, mistral_model, client

GOAL_SETTING_TOOL = {
            "type": "function",
            "function": {
                "name": "goal_setting",
                "description": "Set/ Update the goals for the user based on their input.",
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

def goal_setting(user_query: str) -> Dict[str, str]:
    """
    Set the goals for the user based on their input.
    """
    class GoalSettingAgent(BaseModel):
        calories: str = Field(description = "Give calories")
        protein: str = Field(description = "Give protein in grams")
        carbs: str = Field(description = "Give carbs in grams")
        fat: str = Field(description = "Give fat in grams")

    chat_response = client.chat.parse(
            model=mistral_model,
            messages=[
                {
                    "role": "system",
                    "content": "Extract the goals from the user input."
                },
                {
                    "role": "user",
                    "content": user_query
                },
            ],
            response_format=GoalSettingAgent,
            max_tokens=256,
            temperature=0
        )

    # Extract the goals from the chat response
    goals = json.loads(chat_response.choices[0].message.content)
    global goals_set

    if not goals_set:
      goals_set = goals
    else:
      goals_set.update(goals)

    return chat_response.choices[0].message.content