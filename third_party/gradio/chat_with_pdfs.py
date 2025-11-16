import gradio as gr
from mistralai import Mistral
import numpy as np
import PyPDF2
import faiss

mistral_api_key = "your_api_key"
cli = Mistral(api_key=mistral_api_key)


def get_text_embedding(input: str):
    embeddings_batch_response = cli.embeddings.create(model="mistral-embed", input=input)
    return embeddings_batch_response.data[0].embedding


def rag_pdf(pdfs: list, question: str) -> str:
    chunk_size = 4096
    chunks = []
    for pdf in pdfs:
        chunks += [pdf[i : i + chunk_size] for i in range(0, len(pdf), chunk_size)]

    text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)

    question_embeddings = np.array([get_text_embedding(question)])
    D, I = index.search(question_embeddings, k=4)
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
            messages.append({"role": "user", "content": couple[0]})
            messages.append({"role": "assistant", "content": couple[1]})

    if pdfs:
        pdfs_extracted = []
        for pdf in pdfs:
            reader = PyPDF2.PdfReader(pdf)
            txt = ""
            for page in reader.pages:
                txt += page.extract_text()
            pdfs_extracted.append(txt)

        retrieved_text = rag_pdf(pdfs_extracted, message["text"])
        messages.append({"role": "user", "content": retrieved_text + "\n\n" + message["text"]})
    else:
        messages.append({"role": "user", "content": message["text"]})

    full_response = ""
    for chunk in cli.chat.stream(model="open-mistral-7b", messages=messages, max_tokens=1024):
        full_response += chunk.data.choices[0].delta.content
        yield full_response


app = gr.ChatInterface(fn=ask_mistral, title="Ask Mistral and talk to your PDFs", multimodal=True)
app.launch()
