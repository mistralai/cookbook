# Build NutriSense - An AI-powered food diet companion

A powerful AI-powered food diet companion built with Mistral's agent SDK that helps users with goal setting, logging the food, food recommendations, daily progress, and restaurant recommendations based on their daily diet goals.

[![Food Diet Companion](https://raw.githubusercontent.com/mistralai/cookbook/refs/heads/main/gif/Nutrition_demo.gif)](https://www.youtube.com/watch?v=uEG2z2esl14)

## Features

- 🍽️ **Diet Goal Setting**: Set and update your daily diet goals throughout the day.
- 🥗 **Food Recommendations**: Get personalized meal suggestions based on your diet goals.
- 📝 **Food Logging**: Log your meals as you consume them to keep track of your intake in relation to your goals.
- 📊 **Daily Progress**: View your progress and get a nutritional analysis of your food intake throughout the day.
- 🍴 **Restaurant Recommendations**: Discover restaurants based on your preferences and dietary interests.

## Architecture

![NutriSense Agent Flow Architecture](./assets/nutrisense_architecture.png)

## Example Queries

```bash
# Diet Goal Setting
"Set my calorie goal to 2000 calories, 100g protein, 100g carbs, 100g fibre."

# Food Logging
"Log the food for breakfast - I had 2 eggs, 1 slice of bread, and 1 cup of coffee."

# Meal Recommendation
"Provide recommendations for dinner based on the goals."

# Restaurant Recommendation
"Recommend me some restaurants in paris that serve good grilled chicken."

# Daily Progress
"What is the progress so far for the day?. Provide Nutrition analysis."
```

## Prerequisites

- Python 3.10 or higher
- API Keys:
  - [Mistral AI API key](https://mistral.ai/api-key)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mistralai/cookbook.git
cd mistral/agents/agents_api/food_diet_companion
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
export MISTRAL_API_KEY=<your-mistral-api-key>
```

4. Run the application:
```bash
chainlit run agent.py
```

## Usage

1. Open your browser and navigate to `http://localhost:8000`
2. Start chatting with the assistant using natural language
