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

st.title("Chat with Mistral and your PDFs")

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