import mesop as me
import mesop.labs as mel
from mistralai import Mistral

mistral_api_key = "api_key"
cli = Mistral(api_key=mistral_api_key)


def ask_mistral(message: str, history: list[mel.ChatMessage]):
    messages = [{"role": m.role, "content": m.content} for m in history[:-1]]
    for chunk in cli.chat.stream(model="open-mistral-7b", messages=messages, max_tokens=1024):
        yield chunk.data.choices[0].delta.content


@me.page(title="Talk to Mistral")
def page():
    mel.chat(ask_mistral, title="Ask Mistral", bot_user="Mistral")
