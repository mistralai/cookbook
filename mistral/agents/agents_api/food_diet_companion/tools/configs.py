from mistralai import Mistral
from dotenv import load_dotenv

import os

load_dotenv()

# Mistral model
mistral_model = "mistral-medium-2505"

# Mistral client
client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

# Goals set
goals_set = {}

# Food logs
food_logs = []