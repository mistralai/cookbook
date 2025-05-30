from typing import Dict
from pydantic import BaseModel, Field
import json
from .configs import food_logs, mistral_model, client

FOOD_LOGGING_TOOL = {
            "type": "function",
            "function": {
                "name": "food_log",
                "description": "",
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

def food_logging(user_query: str) -> Dict[str, str]:
    """
    Log food consumption provided by user.
    """
    class FoodLoggingAgent(BaseModel):
        food: str = Field(description = "Give food name")
        quantity: str = Field(description = "Give food quantity")
        calories: str = Field(description = "Give calories in grams")
        protein: str = Field(description = "Give protein in grams")
        carbs: str = Field(description = "Give carbs in grams")
        fat: str = Field(description = "Give fat in grams")

    chat_response = client.chat.parse(
            model=mistral_model,
            messages=[
                {
                    "role": "system",
                    "content": "Understand the food taken by the user and provide nutritional information."
                },
                {
                    "role": "user",
                    "content": user_query
                },
            ],
            response_format=FoodLoggingAgent,
            max_tokens=256,
            temperature=0
        )

    # Extract the food details from the chat response
    food_details = json.loads(chat_response.choices[0].message.content)

    global food_logs
    food_logs.append(food_details)

    return chat_response.choices[0].message.content

