# Chat with your PDF using Mistral and Panel

In this guide, we will introduce the basics of building a chatbot with chat and PDF reading capabilities using `panel`!

**Watch our demo by clicking this image:**

[![Panel Demo](https://img.youtube.com/vi/UpNxJ6wvS2A/0.jpg)](https://www.youtube.com/watch?v=UpNxJ6wvS2A)

## Basic Chat Interface

First, let's implement a simple chat interface. To do this, we will need to import the `panel` and `mistralai` libraries.

```shell
pip install panel mistralai
```

*This demo uses `panel===1.4.4` and `mistralai===0.4.0`*

```py
import panel as pn
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
```

Before proceeding, we must run `pn.extension()` to properly configure `panel`.

```py
pn.extension()
```

Next, create your `MistralClient` instance using your Mistral API key.

```py
mistral_api_key = "your_api_key"
cli = MistralClient(api_key = mistral_api_key)
```

With the client ready, it's time to build the interface. For this, we will use the `ChatInterface` from `panel`!

```py
async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    messages = [ChatMessage(role = "user", content = contents)]
    response = cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 512)
    message = ""
    for chunk in response:
        message += chunk.choices[0].delta.content
        yield message

chat_interface = pn.chat.ChatInterface(callback = callback, callback_user = "Mistral")
chat_interface.servable()
```

In this code, we define a `callback` function that is called every time the user sends a message. This function uses Mistral's models to generate a response.

To run this code, enter `panel serve basic_chat.py` in the console.

<details>
<summary><b>basic_chat.py</b></summary>

```py
import panel as pn
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

pn.extension()

mistral_api_key = "your_api_key"
cli = MistralClient(api_key = mistral_api_key)

async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    messages = [ChatMessage(role = "user", content = contents)]
    response = cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 512)
    message = ""
    for chunk in response:
        message += chunk.choices[0].delta.content
        yield message

chat_interface = pn.chat.ChatInterface(callback = callback, callback_user = "Mistral")
chat_interface.servable()
```

</details>

## Chat History

Currently, the model only has access to the most recent message and does not know about the entire conversation.

To solve this, we need to keep track of the entire chat and provide it to the model. Fortunately, `panel` does this for us!

```py
async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    messages_objects = [w for w in instance.objects]
    messages = [ChatMessage(
        role = "user" if w.user == "User" else "assistant",
        content = w.object
        ) for w in messages_objects]

    response = cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 512)
    message = ""
    for chunk in response:
        message += chunk.choices[0].delta.content
        yield message
```

While we're at it, let's add a welcoming message for the user. We'll need to ignore this message in the callback.

```py
async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    messages_objects = [w for w in instance.objects if w.user != "System"]
    messages = [ChatMessage(
        role="user" if w.user == "User" else "assistant",
        content=w.object
        ) for w in messages_objects]

    response = cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 512)
    message = ""
    for chunk in response:
        message += chunk.choices[0].delta.content
        yield message

chat_interface = pn.chat.ChatInterface(callback = callback, callback_user = "Mistral")
chat_interface.send("Chat with Mistral!", user = "System", respond = False)
chat_interface.servable()
```

We can now have a full conversation with Mistral: `panel serve chat_history.py`

<details>
<summary><b>chat_history.py</b></summary>

```py
import panel as pn
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

pn.extension()

mistral_api_key = "your_api_key"
cli = MistralClient(api_key = mistral_api_key)

async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    messages_objects = [w for w in instance.objects if w.user != "System"]
    messages = [ChatMessage(
        role="user" if w.user == "User" else "assistant",
        content=w.object
        ) for w in messages_objects]

    response = cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 512)
    message = ""
    for chunk in response:
        message += chunk.choices[0].delta.content
        yield message

chat_interface = pn.chat.ChatInterface(callback = callback, callback_user = "Mistral")
chat_interface.send("Chat with Mistral!", user="System", respond=False)
chat_interface.servable()
```
</details>

## Chatting with PDFs

To enable our model to read PDFs, we need to convert the content, extract the text, and then use Mistral's embedding model to retrieve chunks of our document(s) to feed to the model. We will need to implement some basic RAG (Retrieval-Augmented Generation)!

For this task, we will require `faiss`, `PyPDF2`, and other libraries. Let's import them:

```shell
pip install io numpy PyPDF2 faiss
```
**For CPU only please install faiss-cpu instead.**

*This demo uses `numpy===1.26.4`, `PyPDF2===0.4.0` and `faiss-cpu===1.8.0`*

```py
import io
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import numpy as np
import panel as pn
import PyPDF2
import faiss
```

First, we need to add the option to upload files. For this, we will specify the possible inputs for our `ChatInterface`:

```py
chat_interface = pn.chat.ChatInterface(widgets = [pn.widgets.TextInput(),pn.widgets.FileInput(accept = ".pdf")], callback = callback, callback_user = "Mistral")
chat_interface.send("Chat with Mistral and talk to your PDFs!", user = "System", respond = False)
chat_interface.servable()
```

Now the user can both chat and upload a PDF file. Let's handle this new possibility in the callback:

```py
async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    if type(contents) is str:
        messages_objects = [w for w in instance.objects if w.user != "System" and type(w.object) is not pn.chat.message._FileInputMessage]
        messages = [ChatMessage(
            role = "user" if w.user == "User" else "assistant",
            content = w.object
            ) for w in messages_objects]

        pdf_objects = [w for w in instance.objects if w.user != "System" and w not in messages_objects]

        response = cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 1024)
        message = ""
        for chunk in response:
            message += chunk.choices[0].delta.content
            yield message
```

In `pdf_objects`, we will have all previously uploaded PDFs, which will be the documents subject to the RAG.

Let's define a function that will handle all the RAG for us. This function will take the PDFs and the question being asked by the user as input and will return the retrieved chunks concatenated as a string:

```py
def rag_pdf(pdfs: list, question: str) -> str:
    chunk_size = 2048
    chunks = []
    for pdf in pdfs:
        chunks += [pdf[i:i + chunk_size] for i in range(0, len(pdf), chunk_size)]
```

But before continuing, we will need to get the embeddings for all the chunks. Let's quickly create a new function for this:

```py
def get_text_embedding(input_text: str):
    embeddings_batch_response = cli.embeddings(
          model = "mistral-embed",
          input = input_text
      )
    return embeddings_batch_response.data[0].embedding
```

We can now apply the embeddings to the entirety of the chunks, and with `faiss`, we will make a vector store where we will search for the most relevant chunks. Here, we retrieve the best 4 chunks among them:

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

With our RAG function ready, we can implement it in the chat interface. For this, we will use `PyPDF2` to read the PDFs and then use our `rag_pdf` to retrieve the essential text:

```py
async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    if type(contents) is str:
        messages_objects = [w for w in instance.objects if w.user != "System" and type(w.object) is not pn.chat.message._FileInputMessage]
        messages = [ChatMessage(
            role = "user" if w.user == "User" else "assistant",
            content = w.object
            ) for w in messages_objects]

        pdf_objects = [w for w in instance.objects if w.user != "System" and w not in messages_objects]
        if pdf_objects:
            pdfs = []
            for w in pdf_objects:
                reader = PyPDF2.PdfReader(io.BytesIO(w.object.contents))
                txt = ""
                for page in reader.pages:
                    txt += page.extract_text()
                pdfs.append(txt)
            messages[-1].content = rag_pdf(pdfs, contents) + "\n\n" + contents

        response = cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 1024)
        message = ""
        for chunk in response:
            message += chunk.choices[0].delta.content
            yield message
```

If there are PDFs in the chat, it will read them and retrieve the necessary information, which will be concatenated to the original user message.

With this ready, we can now fully chat with Mistral and our PDFs: `panel serve chat_with_pdfs.py`

<details>
<summary><b>chat_with_pdfs.py</b></summary>

```py
import io
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import numpy as np
import panel as pn
import PyPDF2
import faiss

pn.extension()

mistral_api_key = "your_api_key"
cli = MistralClient(api_key = mistral_api_key)

def get_text_embedding(input_text: str):
    embeddings_batch_response = cli.embeddings(
          model = "mistral-embed",
          input = input_text
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
    D, I = index.search(question_embeddings, k = 2)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
    text_retrieved = "\n\n".join(retrieved_chunk)
    return text_retrieved

async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    if type(contents) is str:
        messages_objects = [w for w in instance.objects if w.user != "System" and type(w.object) is not pn.chat.message._FileInputMessage]
        messages = [ChatMessage(
            role = "user" if w.user == "User" else "assistant",
            content = w.object
            ) for w in messages_objects]
        
        pdf_objects = [w for w in instance.objects if w.user != "System" and w not in messages_objects]
        if pdf_objects:
            pdfs = []
            for w in pdf_objects:
                reader = PyPDF2.PdfReader(io.BytesIO(w.object.contents))
                txt = ""
                for page in reader.pages:
                    txt += page.extract_text()
                pdfs.append(txt)
            messages[-1].content = rag_pdf(pdfs, contents) + "\n\n" + contents
        
        response = cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 1024)
        message = ""
        for chunk in response:
            message += chunk.choices[0].delta.content
            yield message

chat_interface = pn.chat.ChatInterface(widgets = [pn.widgets.TextInput(),pn.widgets.FileInput(accept = ".pdf")], callback = callback, callback_user = "Mistral")
chat_interface.send("Chat with Mistral and talk to your PDFs!", user = "System", respond = False)
chat_interface.servable()
```

</details>