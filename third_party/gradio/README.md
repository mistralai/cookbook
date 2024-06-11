# Chat with Your PDF using Mistral and Gradio

In this guide, we will introduce the basics of building a chatbot with chat and PDF reading capabilities using `gradio`!

**Watch our demo by clicking this image:**

[![Gradio Demo](https://img.youtube.com/vi/mrHgm7MOipw/0.jpg)](https://www.youtube.com/watch?v=mrHgm7MOipw)

## Chat Interface

First, let's implement a simple chat interface. To do this, we will need to import the `gradio`, `mistralai` libraries, and `ChatMessage` from `mistralai.models.chat_completion`.

```shell
pip install gradio mistralai
```

*This demo uses `gradio===4.32.2` and `mistralai===0.4.0`*

```py
import gradio as gr
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
```

Next, create your `MistralClient` instance using your Mistral API key.

```py
mistral_api_key = "your_api_key"
cli = MistralClient(api_key = mistral_api_key)
```

To create our interface with `gradio`, we can make use of their `ChatInterface`. It will look something like this:

```py
def ask_mistral(message: str, history: list):
    return "Bot's response."

app = gr.ChatInterface(fn = ask_mistral, title = "Ask Mistral")
app.launch()
```

Now, all we have to do is edit `ask_mistral` so it parses our message and history, calls Mistral's API, and streams the response.

```py
def ask_mistral(message: str, history: list):
    messages = []
    for couple in history:
        messages.append(ChatMessage(role = "user", content = couple[0]))
        messages.append(ChatMessage(role = "assistant", content = couple[1]))
    messages.append(ChatMessage(role = "user", content = message))

    full_response = ""
    for chunk in cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 1024):
        full_response += chunk.choices[0].delta.content
        yield full_response
```

Done! Once ready, you can run the script (`chat.py`)!

<details>
<summary><b>chat.py</b></summary>

```py
import gradio as gr
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

mistral_api_key = "your_api_key"
cli = MistralClient(api_key = mistral_api_key)

def ask_mistral(message: str, history: list):
    messages = []
    for couple in history:
        messages.append(ChatMessage(role = "user", content = couple[0]))
        messages.append(ChatMessage(role = "assistant", content = couple[1]))
    messages.append(ChatMessage(role = "user", content = message))

    full_response = ""
    for chunk in cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 1024):
        full_response += chunk.choices[0].delta.content
        yield full_response

app = gr.ChatInterface(fn = ask_mistral, title = "Ask Mistral")
app.launch()
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
import gradio as gr
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import numpy as np
import PyPDF2
import faiss
```

For our interface to allow the uploading of files, we need to toggle multimodality on our `ChatInterface`.

```py
app = gr.ChatInterface(fn = ask_mistral, title = "Ask Mistral and talk to your PDFs", multimodal = True)
app.launch()
```

Now, our interface will also accept files. The next step is to handle them and filter out the PDF files from the messages.

```py
def ask_mistral(message: str, history: list):
    messages = []
    pdfs = message["files"]
    for couple in history:
        if type(couple[0]) is tuple:
            pdfs += couple[0]
        else:
            messages.append(ChatMessage(role = "user", content = couple[0]))
            messages.append(ChatMessage(role = "assistant", content = couple[1]))

    messages.append(ChatMessage(role = "user", content = message["text"]))

    full_response = ""
    for chunk in cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 1024):
        full_response += chunk.choices[0].delta.content
        yield full_response
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

In this function, we cut the PDF files into chunks of equal sizes, get their embeddings, and apply some vector search with `faiss` to retrieve the best 4 chunks. The next and last step will be to read the PDF files themselves with `PyPDF2` and integrate them with the model:

```py
def ask_mistral(message: str, history: list):
    messages = []
    pdfs = message["files"]
    for couple in history:
        if type(couple[0]) is tuple:
            pdfs += couple[0]
        else:
            messages.append(ChatMessage(role = "user", content = couple[0]))
            messages.append(ChatMessage(role = "assistant", content = couple[1]))

    if pdfs:
        pdfs_extracted = []
        for pdf in pdfs:
            reader = PyPDF2.PdfReader(pdf)
            txt = ""
            for page in reader.pages:
                txt += page.extract_text()
            pdfs_extracted.append(txt)

        retrieved_text = rag_pdf(pdfs_extracted, message["text"])
        messages.append(ChatMessage(role = "user", content = retrieved_text + "\n\n" + message["text"]))
    else:
        messages.append(ChatMessage(role = "user", content = message["text"]))

    full_response = ""
    for chunk in cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 1024):
        full_response += chunk.choices[0].delta.content
        yield full_response
```

With this, we are ready to go! We can run our script `chat_with_pdfs.py`.

<details>
<summary><b>chat_with_pdfs.py</b></summary>

```py
import gradio as gr
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import numpy as np
import PyPDF2
import faiss

mistral_api_key = "your_api_key"
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

def ask_mistral(message: str, history: list):
    messages = []
    pdfs = message["files"]
    for couple in history:
        if type(couple[0]) is tuple:
            pdfs += couple[0]
        else:
            messages.append(ChatMessage(role= "user", content = couple[0]))
            messages.append(ChatMessage(role= "assistant", content = couple[1]))

    if pdfs:
        pdfs_extracted = []
        for pdf in pdfs:
            reader = PyPDF2.PdfReader(pdf)
            txt = ""
            for page in reader.pages:
                txt += page.extract_text()
            pdfs_extracted.append(txt)

        retrieved_text = rag_pdf(pdfs_extracted, message["text"])
        messages.append(ChatMessage(role = "user", content = retrieved_text + "\n\n" + message["text"]))
    else:
        messages.append(ChatMessage(role = "user", content = message["text"]))

    full_response = ""
    for chunk in cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 1024):
        full_response += chunk.choices[0].delta.content
        yield full_response

app = gr.ChatInterface(fn = ask_mistral, title = "Ask Mistral and talk to your PDFs", multimodal = True)
app.launch()
```

</details>