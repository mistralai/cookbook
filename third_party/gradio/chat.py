import gradio as gr
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

mistral_api_key = "your_api_key"
cli = MistralClient(api_key = mistral_api_key)

def ask_mistral(message: str, history: list):
    messages = []
    for couple in history:
        messages.append(ChatMessage(role= "user", content = couple[0]))
        messages.append(ChatMessage(role= "assistant", content = couple[1]))

    messages.append(ChatMessage(role = "user", content = message))
    full_response = ""
    for chunk in cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 1024):
        full_response += chunk.choices[0].delta.content
        yield full_response

app = gr.ChatInterface(fn = ask_mistral, title = "Ask Mistral")
app.launch()