import streamlit as st
from mistralai.client import MistralClient

mistral_api_key = "your_api_key"
cli = MistralClient(api_key = mistral_api_key)

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