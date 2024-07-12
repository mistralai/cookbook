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
