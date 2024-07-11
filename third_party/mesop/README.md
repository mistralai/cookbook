# Chat with Your PDF using Mistral and Mesop

In this guide, we will introduce the basics of building a chatbot with chat and PDF reading capabilities using `mesop`!

## Chat Interface

First, let's implement a simple chat interface. To do this, we will need to import the `mesop`, `mesop.labs`, `mistralai` libraries, and `ChatMessage` from `mistralai.models.chat_completion`.

```shell
pip install mesop mistralai
```

*This demo uses `mesop===0.9.3` and `mistralai===0.4.0`*

```py
import mesop as me
import mesop.labs as mel
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
```

Next, create your `MistralClient` instance using your Mistral API key.

```py
mistral_api_key = "api_key"
cli = MistralClient(api_key = mistral_api_key)
```

To create our interface with `mesop`, we can make use of their `chat` function. It will look something like this:

```py
def ask_mistral(message: str, history: list[mel.ChatMessage]):
    messages = [ChatMessage(role=m.role, content=m.content) for m in history[:-1]]
    for chunk in cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 1024):
        yield chunk.choices[0].delta.content

@me.page(title="Talk to Mistral")
def page():
  mel.chat(ask_mistral, title="Ask Mistral", bot_user="Mistral")
```

Now, all we have to do is run the command `mesop chat.py`!

<details>
<summary><b>chat.py</b></summary>

```py
import mesop as me
import mesop.labs as mel
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

mistral_api_key = "api_key"
cli = MistralClient(api_key = mistral_api_key)

def ask_mistral(message: str, history: list[mel.ChatMessage]):
    messages = [ChatMessage(role=m.role, content=m.content) for m in history[:-1]]
    for chunk in cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 1024):
        yield chunk.choices[0].delta.content

@me.page(title="Talk to Mistral")
def page():
  mel.chat(ask_mistral, title="Ask Mistral", bot_user="Mistral")
```

</details>

## Chatting with PDFs

To enable our model to read PDFs, we need to convert the content, extract the text, and then use Mistral's embedding model to retrieve chunks of our document(s) to feed to the model. We will need to implement some basic RAG (Retrieval-Augmented Generation)!

For this task, we will require `faiss`, `PyPDF2`, and other libraries. Let's import them:

```shell
pip install numpy PyPDF2 faiss
```
**For CPU only please install faiss-cpu instead.**

*This demo uses `numpy===1.26.4`, `PyPDF2===0.4.0` and `faiss-cpu===1.8.0`*

```py
import io
import mesop as me
import mesop.labs as mel
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import numpy as np
import PyPDF2
import faiss
```

For our interface to allow the uploading of files, we need to add an uploader to our `page` function.

```py
@me.page(title="Talk to Mistral")
def page():
    with me.box(style=me.Style(height = "100%", display="flex", flex_direction="column", align_items="center",padding=me.Padding(top = 0, left = 30, right = 30, bottom = 0))):
        with me.box(style=me.Style(padding=me.Padding(top = 16), position="fixed")):
            me.uploader(
                label="Upload PDF",
                accepted_file_types=["file/pdf"],
                on_upload=handle_upload,
            )
        with me.box(style=me.Style(width="100%")):
            mel.chat(ask_mistral, title="Ask Mistral", bot_user="Mistral")
```

Now, our interface will also accept files. The next step is to handle them and extract the text from the PDF files.

```py
@me.stateclass
class State:
  content: str
  
def handle_upload(event: me.UploadEvent):
    state = me.state(State)
    reader = PyPDF2.PdfReader(io.BytesIO(event.file.getvalue()))
    txt = ""
    for page in reader.pages:
        txt += page.extract_text()
    state.content = txt
```

We are ready to read the PDF files and implement some RAG. For this, we will need to make a function that retrieves the relevant chunks of text from the PDFs concatenated as a single string. For that, we will make use of Mistral's embeddings. Let's quickly design a function that will convert text to the embeddings:

```py
def get_text_embedding(input: str):
    embeddings_batch_response = cli.embeddings(
          model = "mistral-embed",
          input = input
      )
    return embeddings_batch_response.data[0].embedding
```

And now, we can make `rag_pdf` that will handle all the RAG and retrieve the proper chunks:

```py
def rag_pdf(pdfs: list, question: str) -> str:
    chunk_size = 4096
    chunks = []
    for pdf in pdfs:
        chunks += [pdf[i:i + chunk_size] for i in range(0, len(pdf), chunk_size)]

    text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)

    question_embeddings = np.array([get_text_embedding(question)])
    D, I = index.search(question_embeddings, k = 4)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
    text_retrieved = "\n\n".join(retrieved_chunk)
    return text_retrieved
```

In this function, we cut the PDF files into chunks of equal sizes, get their embeddings, and apply some vector search with `faiss` to retrieve the best 4 chunks. The next and last step will be to integrate them with the model:

```py
def ask_mistral(message: str, history: list[mel.ChatMessage]):
    messages = [ChatMessage(role=m.role, content=m.content) for m in history[:-1]]

    state = me.state(State)
    if state.content:
        retrieved_text = rag_pdf([state.content], message)
        messages[-1] = ChatMessage(role = "user", content = retrieved_text + "\n\n" +messages[-1].content)

    for chunk in cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 1024):
        yield chunk.choices[0].delta.content
```

With this, we are ready to go! We can run our script with the command `mesop chat_with_pdfs.py`.

<details>
<summary><b>chat_with_pdfs.py</b></summary>

```py
import io
import mesop as me
import mesop.labs as mel
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import numpy as np
import PyPDF2
import faiss

mistral_api_key = "api_key"
cli = MistralClient(api_key = mistral_api_key)

def get_text_embedding(input: str):
    embeddings_batch_response = cli.embeddings(
          model = "mistral-embed",
          input = input
      )
    return embeddings_batch_response.data[0].embedding

def rag_pdf(pdfs: list, question: str) -> str:
    chunk_size = 4096
    chunks = []
    for pdf in pdfs:
        chunks += [pdf[i:i + chunk_size] for i in range(0, len(pdf), chunk_size)]

    text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)

    question_embeddings = np.array([get_text_embedding(question)])
    D, I = index.search(question_embeddings, k = 4)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
    text_retrieved = "\n\n".join(retrieved_chunk)
    return text_retrieved

def ask_mistral(message: str, history: list[mel.ChatMessage]):
    messages = [ChatMessage(role=m.role, content=m.content) for m in history[:-1]]

    state = me.state(State)
    if state.content:
        retrieved_text = rag_pdf([state.content], message)
        messages[-1] = ChatMessage(role = "user", content = retrieved_text + "\n\n" +messages[-1].content)

    for chunk in cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 1024):
        yield chunk.choices[0].delta.content

@me.stateclass
class State:
  content: str

def handle_upload(event: me.UploadEvent):
    state = me.state(State)
    reader = PyPDF2.PdfReader(io.BytesIO(event.file.getvalue()))
    txt = ""
    for page in reader.pages:
        txt += page.extract_text()
    state.content = txt

@me.page(title="Talk to Mistral")
def page():
    with me.box(style=me.Style(height = "100%", display="flex", flex_direction="column", align_items="center",padding=me.Padding(top = 0, left = 30, right = 30, bottom = 0))):
        with me.box(style=me.Style(padding=me.Padding(top = 16), position="fixed")):
            me.uploader(
                label="Upload PDF",
                accepted_file_types=["file/pdf"],
                on_upload=handle_upload,
            )
        with me.box(style=me.Style(width="100%")):
            mel.chat(ask_mistral, title="Ask Mistral", bot_user="Mistral")
```

</details>
