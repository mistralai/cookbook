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
          model="mistral-embed",
          input=input_text
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

async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    if type(contents) is str:
        messages_objects = [w for w in instance.objects if w.user != "System" and type(w.object) is not pn.chat.message._FileInputMessage]
        messages = [ChatMessage(
            role="user" if w.user == "User" else "assistant",
            content=w.object
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
        
        response = cli.chat_stream(model = "open-mistral-7b", messages = messages, max_tokens = 1024, temperature = 0.7)
        message = ""
        for chunk in response:
            message += chunk.choices[0].delta.content
            yield message

chat_interface = pn.chat.ChatInterface(widgets = [pn.widgets.TextInput(),pn.widgets.FileInput(accept = ".pdf")], callback = callback, callback_user = "Mistral")
chat_interface.send("Chat with Mistral and talk to your PDFs!", user = "System", respond = False)
chat_interface.servable()