from .configs import goals_set, food_logs, mistral_model, client

FOOD_RECOMMENDATIONS_TOOL = {
            "type": "function",
            "function": {
                "name": "food_recommendations",
                "description": "Provide meal recommendations for the user.",
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

def food_recommendations(user_query):
    """Provide personalized meal and nutrition recommendations"""
    global goals_set, food_logs

    prompt = f"""
    You are a nutrition recommendation assistant. Provide personalized food and meal suggestions.

    User goals: {goals_set}
    Today's logs so far: {food_logs}

    Based on their goals and what they've already consumed today:
    1. Suggest appropriate meals or foods that would help them meet their nutritional targets
    2. Consider what they've already eaten to ensure balanced nutrition
    3. Provide recommendations for the meals for which user asked.
    4. Answer should be very short and concise


    If they haven't set goals yet, provide general healthy recommendations and encourage them to set specific targets.
    If they haven't logged any food today, assume they're starting fresh and provide full-day meal suggestions.

    User Query: {user_query}
    """

    response = client.chat.complete(
            model=mistral_model,
            messages=[{"role": "user", "content": prompt}]
        )
    full_response = response.choices[0].message.content

    return full_response