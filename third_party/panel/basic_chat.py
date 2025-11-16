import panel as pn
from mistralai import Mistral

pn.extension()

mistral_api_key = "your_api_key"
cli = Mistral(api_key=mistral_api_key)


async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    messages = [{"role": "user", "content": contents}]
    response = cli.chat.stream(model="open-mistral-7b", messages=messages, max_tokens=1024)
    message = ""
    for chunk in response:
        message += chunk.data.choices[0].delta.content
        yield message


chat_interface = pn.chat.ChatInterface(callback=callback, callback_user="Mistral")
chat_interface.servable()
