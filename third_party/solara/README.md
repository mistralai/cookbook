# Chat with Your PDF using Mistral and Solara

*Author: Alonso Silva Allende (Nokia Bell Labs), GitHub handle: [alonsosilvaallende](https://github.com/alonsosilvaallende/)*

In this guide, we introduce the basics of building a chatbot with chat and PDF reading capabilities using `solara`

## Chat Interface

Let's implement a simple chat interface. To do this, we need to import `solara` and `mistralai` libraries.

```shell
pip install solara mistralai
```

*This demo uses `solara===1.41.0` and `mistralai===1.2.3`*

```py
import solara as sl
from mistralai import Mistral
```

Create your `client` using your Mistral API key.

```py
mistral_api_key = "your_api_key"
client = Mistral(api_key = mistral_api_key)
```

Let's initialize a reactive variable where all messages will be stored.

```py
from typing import List
from typing_extensions import TypedDict

class MessageDict(TypedDict):
    role: str
    content: str

messages: sl.Reactive[List[MessageDict]] = sl.reactive([])
```

Given a list of messages (for the moment empty but not for long), we query Mistral and retrieve the response. To make the interaction smooth, we handle it by streaming the response. For this, we define a generator.

```py
def response_generator(messages):
    response = client.chat.stream(model="open-mistral-7b", messages=messages, max_tokens=1024)
    for chunk in response:
        yield chunk.data.choices[0].delta.content
```

We stream the response by displaying each chunk as it is received.

```py
def add_chunk_to_ai_message(chunk: str):
    messages.value = [
        *messages.value[:-1],
        {
            "role": "assistant",
            "content": messages.value[-1]["content"] + chunk,
        },
    ]
```

Given a list of messages, we display them on the screen:

```py
@sl.component
def Page():
    with sl.lab.ChatBox():
        for item in messages.value:
            with sl.lab.ChatMessage(
                user=item["role"] == "user",
                name="User" if item["role"] == "user" else "Assistant"
            ):
                sl.Markdown(item["content"])
```

The following step is to retrieve the input from the user and store it in the list of messages. For this, we will use `ChatInput` from `solara`

```py
        def send(user_message):
            messages.value = [*messages.value, {"role": "user", "content": user_message}]
        sl.lab.ChatInput(send_callback=send)
```

We need to handle a streamed response. Therefore we create a task which will be activated by a change on the number of user messages.
```py
        user_message_count = len([m for m in messages.value if m["role"] == "user"])
        def response(messages):
            messages.value = [*messages.value, {"role": "assistant", "content": ""}]
            for chunk in response_generator(messages.value[:-1]):
                add_chunk_to_ai_message(chunk)
        def result():
            if messages.value != []:
                response(messages)
        result = sl.lab.use_task(result, dependencies=[user_message_count])
```

That's it! An interface where you can chat with Mistral's models. I added some optional styling below.

To run this code, enter `solara run chat.py` in the console. Alternatively, you can modify it directly in [PyCafe](https://py.cafe/alonsosilvaallende/solara-mistral-ai-chat).

<details>
<summary><b>chat.py</b></summary>

```py
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
```

</details>

## Chatting with PDFs

To enable our model to read PDFs, we need to convert the content, extract the text, and then use Mistral's embedding model to retrieve chunks of our document(s) to feed to the model. We need to implement some basic RAG (Retrieval-Augmented Generation)!

For this task, we require `faiss` and `PyPDF2`. Let's import them:
```py
pip install PyPDF2 faiss
```

**For CPU only please install faiss-cpu instead.**

This demo uses `PyPDF2===3.0.1` and `faiss-cpu===1.8.0`
```py
import io
import solara as sl
from mistralai import Mistral
import numpy as np
import PyPDF2
import faiss
```

Now, we need to add the possibility to upload PDF files. For this, let's use `FileDropMultiple` from `solara`. The PDFs will then be stored in a new reactive variable:

```py
from solara.components.file_drop import FileInfo

content, set_content = sl.use_state(cast(List[bytes], []))

def on_file(files: List[FileInfo]):
    set_content([file["file_obj"].read() for file in files])

sl.FileDropMultiple(
    label="Drag and drop your PDF file(s) here.",
    on_file=on_file,
    lazy=True,
)
```

The PDFs are stored, but as they are, we just have a large amount of bytes. To be able to chat with the PDF, we will need to extract the text:
```py
txt = sl.use_reactive(cast(List[str], []))

def get_text():
    txt_all = []
    for _content in content:
        bytes_io = io.BytesIO(_content)
        reader = PyPDF2.PdfReader(bytes_io)
        txt_aux = ""
        for page in reader.pages:
            txt_aux += page.extract_text()
        txt_all.append(txt_aux)
    return txt_all

if content:
    sl.Info("File(s) has been uploaded. Showing the beginning of the file(s)...")
    result: Task[List[str]] = use_task(get_text, dependencies=[content])
    if result.finished:
        txt.value = result.value
    sl.ProgressLinear(result.pending)
    for text in txt.value:
        sl.Markdown(f"{text[:100]}")
```

Now that we have the texts, let's use Mistral's embeddings to retrieve the relevant chunks. First, let's define a function that converts text to embeddings with Mistral:

```py
def get_text_embedding(input_text: str):
    embeddings_batch_response = client.embeddings(
          model = "mistral-embed",
          input = input_text
      )
    return embeddings_batch_response.data[0].embedding
```

Next, we can declare a function that will handle all the retrieval part. This step will make use of `faiss` for the vector store and the previously created `get_text_embedding function`. This will cut the different files into chunks, create the embeddings, and retrieve the best 4 chunks among them, which will then be concatenated into a single string:

```py
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
```

Finally, we edit `response_generator` to implement our new RAG with the files! This function, when there are PDFs, will extract the text with PyPDF2 and make use of `rag_pdf` to retrieve the relevant data. It will only then send the request to the model:

```py
def response_generator(messages: list, txt: List[str]):
    response = client.chat.stream(
        model = "open-mistral-7b",
        messages = messages[:-1] + [{"role":"user","content": rag_pdf(txt, messages[-1]["content"]) + "\n\n" + messages[-1]["content"]}],
        max_tokens = 1024
    )
    for chunk in response:
        yield chunk.data.choices[0].delta.content
```

And everything is done! Now we can run our new interface with `solara run chat_with_pdfs.py`

<details>
<summary><b>chat_with_pdfs.py</b></summary>

```py
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
```

</details>
