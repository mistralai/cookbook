# Chat with Your PDF using Mistral and Streamlit

In this guide, we will introduce the basics of building a chatbot with chat and PDF reading capabilities using `streamlit`!

**Watch our demo by clicking this image:**

[![Panel Demo](https://img.youtube.com/vi/VGSAA-d_Sqo/0.jpg)](https://www.youtube.com/watch?v=VGSAA-d_Sqo)

## Chat Interface

First, let's implement a simple chat interface. To do this, we will need to import the `streamlit` and `mistralai` libraries.

```shell
pip install streamlit mistralai
```

*This demo uses `streamlit===1.35.0` and `mistralai===0.4.0`*

```py
import streamlit as st
from mistralai.client import MistralClient
```

Next, create your `MistralClient` instance using your Mistral API key.

```py
mistral_api_key = "your_api_key"
cli = MistralClient(api_key = mistral_api_key)
```

Now, we will initialize a session variable where all messages will be stored and display them on the screen.

```py
st.title("Chat with Mistral and your PDFs")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
```

The following step is to retrieve the input from the user and store it in the list of messages. For this, we will use `chat_input` from `streamlit`!

```py
if prompt := st.chat_input("Talk to Mistral!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
```

All that's left is to query Mistral and retrieve the response. To make the interaction smooth, we will handle it by streaming the response. For this, `streamlit` has `write_stream`, which accepts a generator. Let's define a generator!

```py
def ask_mistral(messages: list):
    resp = cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 1024)
    for chunk in resp:
        yield chunk.choices[0].delta.content
```

With everything set, all we need to do is retrieve the response from the model and save it in the session.

```py
if prompt := st.chat_input("Talk to Mistral!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response_generator = ask_mistral(st.session_state.messages)
        response = st.write_stream(response_generator)

    st.session_state.messages.append({"role": "assistant", "content": response})
```

There you go! An interface where you can chat with Mistral's models.

To run this code, enter `streamlit run chat.py` in the console.

<details>
<summary><b>chat.py</b></summary>

```py
import streamlit as st
from mistralai.client import MistralClient

mistral_api_key = "your_api_key"
cli = MistralClient(api_key=mistral_api_key)

st.title("Chat with Mistral")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def ask_mistral(messages: list):
    resp = cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 1024)
    for chunk in resp:
        yield chunk.choices[0].delta.content

if prompt := st.chat_input("Talk to Mistral!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response_generator = ask_mistral(st.session_state.messages)
        response = st.write_stream(response_generator)

    st.session_state.messages.append({"role": "assistant", "content": response})
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
import streamlit as st
from mistralai.client import MistralClient
import numpy as np
import PyPDF2
import faiss
```

Now, we need to add the possibility to upload PDF files. For this, let's use `file_uploader` from `streamlit`. The PDF will then be stored in a new session variable:

```py
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.pdfs = []

# The rest of the code...

uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
if uploaded_file is not None:
    bytes_io = io.BytesIO(uploaded_file.getvalue())
    st.session_state.pdfs.append(bytes_io)
```

The PDFs are stored, but as they are, we just have a large amount of bytes. To be able to chat with the PDF, we will need to extract the text and use Mistral's embeddings to retrieve the relevant chunks.

First, let's define a function that converts text to embeddings with Mistral:

```py
def get_text_embedding(input_text: str):
    embeddings_batch_response = cli.embeddings(
          model = "mistral-embed",
          input = input_text
      )
    return embeddings_batch_response.data[0].embedding
```

Next, we can declare a function that will handle all the retrieval part. This step will make use of `faiss` for the vector store and the previously created `get_text_embedding` function. This will cut the different files into chunks, create the embeddings, and retrieve the best 4 chunks among them, which will then be concatenated into a single string:

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

Finally, we edit `ask_mistral` to implement our new RAG with the files! This function, when there are PDFs, will extract the text with `PyPDF2` and make use of `rag_pdf` to retrieve the relevant data. It will only then send the request to the model:

```py
def ask_mistral(messages: list, pdfs_bytes: list):
    if pdfs_bytes:
        pdfs = []
        for pdf in pdfs_bytes:
            reader = PyPDF2.PdfReader(pdf)
            txt = ""
            for page in reader.pages:
                txt += page.extract_text()
            pdfs.append(txt)
        messages[-1]["content"] = rag_pdf(pdfs, messages[-1]["content"]) + "\n\n" + messages[-1]["content"]

    resp = cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 1024)
    for chunk in resp:
        yield chunk.choices[0].delta.content

# Don't forget to add the new argument 'pdfs_bytes = st.session_state.pdfs' when you call this function.
```

And everything is done! Now we can run our new interface with `streamlit run chat_with_pdfs.py`.

<details>
<summary><b>chat_with_pdfs.py</b></summary>

```py
import io
import streamlit as st
from mistralai.client import MistralClient
import numpy as np
import PyPDF2
import faiss

mistral_api_key = "your_api_key"
cli = MistralClient(api_key = mistral_api_key)

def get_text_embedding(input: str):
    embeddings_batch_response = cli.embeddings(
          model="mistral-embed",
          input=input
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

st.title("Chat with Mistral")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.pdfs = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def ask_mistral(messages: list, pdfs_bytes: list):
    if pdfs_bytes:
        pdfs = []
        for pdf in pdfs_bytes:
            reader = PyPDF2.PdfReader(pdf)
            txt = ""
            for page in reader.pages:
                txt += page.extract_text()
            pdfs.append(txt)
        messages[-1]["content"] = rag_pdf(pdfs, messages[-1]["content"]) + "\n\n" + messages[-1]["content"]
        
    resp = cli.chat_stream(model="open-mistral-7b", messages = messages, max_tokens = 1024)
    for chunk in resp:
        yield chunk.choices[0].delta.content

if prompt := st.chat_input("Talk to Mistral!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response_generator = ask_mistral(st.session_state.messages, st.session_state.pdfs)
        response = st.write_stream(response_generator)

    st.session_state.messages.append({"role": "assistant", "content": response})

uploaded_file = st.file_uploader("Choose a file", type = ["pdf"])
if uploaded_file is not None:
    bytes_io = io.BytesIO(uploaded_file.getvalue())
    st.session_state.pdfs.append(bytes_io)
```

</details>