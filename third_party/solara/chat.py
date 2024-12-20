import solara as sl
from mistralai import Mistral

mistral_api_key = "your_api_key"
client = Mistral(api_key=mistral_api_key)

from typing import List
from typing_extensions import TypedDict

class MessageDict(TypedDict):
    role: str
    content: str

messages: sl.Reactive[List[MessageDict]] = sl.reactive([])

def response_generator(messages):
    response = client.chat.stream(model="open-mistral-7b", messages=messages, max_tokens=1024)
    for chunk in response:
        yield chunk.data.choices[0].delta.content

def add_chunk_to_ai_message(chunk: str):
    messages.value = [
        *messages.value[:-1],
        {
            "role": "assistant",
            "content": messages.value[-1]["content"] + chunk,
        },
    ]

@sl.component
def Page():
    user_message_count = len([m for m in messages.value if m["role"] == "user"])
    def send(user_message):
        messages.value = [*messages.value, {"role": "user", "content": user_message}]
    def response(messages):
        messages.value = [*messages.value, {"role": "assistant", "content": ""}]
        for chunk in response_generator(messages.value[:-1]):
            add_chunk_to_ai_message(chunk)
    def result():
        if messages.value != []:
            response(messages)
    result = sl.lab.use_task(result, dependencies=[user_message_count])
    with sl.Column(align="center"):
        with sl.lab.ChatBox(style={"position": "fixed", "overflow-y": "scroll","scrollbar-width": "none", "-ms-overflow-style": "none", "top": "0", "bottom": "10rem", "width": "60%"}):
            for item in messages.value:
                with sl.lab.ChatMessage(
                    user=item["role"] == "user",
                    name="User" if item["role"] == "user" else "Assistant"
                ):
                    sl.Markdown(item["content"])
        sl.lab.ChatInput(send_callback=send, style={"position": "fixed", "bottom": "3rem", "width": "70%"})
