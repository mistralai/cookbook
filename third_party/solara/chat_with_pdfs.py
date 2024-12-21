import io
import solara as sl
from mistralai import Mistral
import numpy as np
import PyPDF2
import faiss
from solara.components.file_drop import FileInfo
from solara.lab import use_task, Task
from typing import List, cast
from typing_extensions import TypedDict

mistral_api_key = "your_api_key"
client = Mistral(api_key=mistral_api_key)

def get_text_embedding(input_text: str):
    embeddings_batch_response = client.embeddings(
          model = "mistral-embed",
          input = input_text
      )
    return embeddings_batch_response.data[0].embedding

def rag_pdf(txt: List[str], question: str) -> str:
    chunk_size = 1024
    chunks = []
    for _txt in txt:
        chunks += [_txt[i:i + chunk_size] for i in range(0, len(_txt), chunk_size)]

    text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)

    question_embeddings = np.array([get_text_embedding(question)])
    D, I = index.search(question_embeddings, k = 3)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
    text_retrieved = "\n\n".join(retrieved_chunk)
    return text_retrieved

class MessageDict(TypedDict):
    role: str
    content: str

messages: sl.Reactive[List[MessageDict]] = sl.reactive([])

def response_generator(messages: list, txt: List[str]):
    response = client.chat.stream(
        model = "open-mistral-7b", 
        messages = messages[:-1] + [{"role":"user","content": rag_pdf(txt, messages[-1]["content"]) + "\n\n" + messages[-1]["content"]}],
        max_tokens = 1024
    )
    for chunk in response:
        yield chunk.choices[0].delta.content

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
    txt = sl.use_reactive(cast(List[str], []))
    with sl.Sidebar():

        def on_file(files: List[FileInfo]):
            get_text([file["data"] for file in files])

        @sl.lab.task
        def get_text(pdf_content):
            txt_all = []
            for _content in pdf_content:
                bytes_io = io.BytesIO(_content)
                reader = PyPDF2.PdfReader(bytes_io)
                txt_aux = ""
                for page in reader.pages:
                    txt_aux += page.extract_text()
                txt_all.append(txt_aux)
            return txt_all


        sl.FileDropMultiple(
            label="Drag and drop your PDF file(s) here.",
            on_file=on_file,
            lazy=True,
        )

        sl.ProgressLinear(get_text.pending)
        if get_text.value:
            sl.Info("File(s) has been uploaded. Showing the beginning of the file(s)...")
            for text in get_text.value:
                sl.Markdown(f"{text[:100]}")

    user_message_count = len([m for m in messages.value if m["role"] == "user"])
    def send(user_message):
        messages.value = [*messages.value, {"role": "user", "content": user_message}]
    def response(messages):
        messages.value = [*messages.value, {"role": "assistant", "content": ""}]
        for chunk in response_generator(messages.value[:-1], txt=txt.value):
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
            sl.lab.ChatInput(send_callback=send, style={"position": "fixed", "bottom": "3rem", "width": "60%"})
