import functools
import json
import os

from mistralai.client import MistralClient
from openai import OpenAI

from myfuncs import list_pods

api_key = os.environ.get("MISTRAL_API_KEY")
             
online_model = "mistral-small-latest"
offline_model = "mistral" # ollama naming convention

tools = [
    {
        "type": "function",
        "function": {
            "name": "list_pods",
            "description": "Get the list of all Kubernetes pods and their status in a given namespace",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "The name of the namespace to look into",
                    },
                },
                "required": ["namespace"],
            },
        }
    },
]

callables = {"list_pods": list_pods}

user_input = input("😺 Hello! How can I help you?\n")

# Retrieve user input then generate function inputs with mistral-small
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": f"Execute the following task on your K8S cluster: {user_input}"})
online_client = MistralClient(api_key=api_key)
resp_tool= online_client.chat(model=online_model, messages=messages, tools=tools, tool_choice="any")
print(f"⏳ Using online model {online_model} to generate function inputs, un instant svp...")
tool_call = resp_tool.choices[0].message.tool_calls[0]
function_name = tool_call.function.name
function_params = json.loads(tool_call.function.arguments)
print(f"😎 Switching to offline execution and calling ollama's {offline_model}. C'est parti!\n\n")

# Run the function
partial_func = functools.partial(callables[function_name], **function_params)
out = partial_func()

# Format the function output with ollama-mistral (7b)
local_client = OpenAI(base_url = "http://127.0.0.1:11434/v1", api_key='ollama')
response = local_client.chat.completions.create(stream=True, model=offline_model, messages=[
    {"role": "system", "content": "You are a master of Kubernetes who likes friendly French-related jokes."},
    {"role": "user", "content": f"""
     Here is a list of K8S pods in a JSON format:\n {out}
     Transform it into a bullet-point-like list in idiomatic English and add a few French-related jokes.
        """}])

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="") 

