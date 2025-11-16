import gradio as gr
from mistralai import Mistral

mistral_api_key = "your_api_key"
cli = Mistral(api_key=mistral_api_key)


def ask_mistral(message: str, history: list):
    messages = []
    for couple in history:
        messages.append({"role": "user", "content": couple[0]})
        messages.append({"role": "assistant", "content": couple[1]})

    messages.append({"role": "user", "content": message})
    full_response = ""
    for chunk in cli.chat.stream(model="open-mistral-7b", messages=messages, max_tokens=1024):
        full_response += chunk.data.choices[0].delta.content
        yield full_response


app = gr.ChatInterface(fn=ask_mistral, title="Ask Mistral")
app.launch()
