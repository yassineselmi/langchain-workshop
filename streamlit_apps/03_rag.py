"""
RAG (Retrieval-Augmented Generation) is a model that combines a retriever and a generator to answer questions.
"""

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader



st.set_page_config(page_title="RAG ChatBot", page_icon="💬")
st.title("💬 Langchain: Chat with your documents")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM and Langchain")

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="chatbot_api_key", type="password"
    )
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    if not (openai_api_key.startswith("sk-")):
        st.warning("Please enter your OpenAI API key!", icon="⚠")

    st.subheader("Models and parameters")
    temperature = st.sidebar.slider(
        "temperature", min_value=0.01, max_value=1.0, value=0.1, step=0.01
    )

    st.subheader("Conversation memory")
    memory_buffer_size = st.sidebar.slider(
        "Memory buffer size", min_value=1, max_value=100, value=10, step=1
    )

    st.subheader("Private files")
    uploaded_file = st.file_uploader("Choose a PDF file",  type='pdf')


if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()


if not uploaded_file:
    st.info("Please upload a PDF file to continue.")
    st.stop()


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def load_documents(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    content = ""
    for page in pdf_reader.pages:
        content += page.extract_text()
    return content

# 1. Setup LLM
llm = ChatOpenAI(
    temperature=temperature,
    openai_api_key=openai_api_key,
    model="gpt-3.5-turbo",
    streaming=True,
)


# 2. Setup Memory
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferWindowMemory(
    k=memory_buffer_size, return_messages=True, chat_memory=msgs
)

# 3. Setup the embedding + text splitter + retriever

# embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
embed_model = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")

# Load document if file is uploaded
content = load_documents(uploaded_file)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = text_splitter.split_text(text=content)
vectorstore = Chroma.from_texts(chunks, embed_model)

# 3. Setup Conversational Chain
rag_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

# Streamlit - User input handler
if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        response = rag_chain.run(user_query, callbacks=[stream_handler])
        