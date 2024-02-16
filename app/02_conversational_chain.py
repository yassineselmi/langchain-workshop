import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler


st.set_page_config(page_title="ConversationChain ChatBot", page_icon="💬")
st.title("💬 Langchain: Simple Chatbot with ConversationChain")
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

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


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

# 3. Setup Conversational Chain
convo_chain = ConversationChain(llm=llm, memory=memory)

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
        response = convo_chain.run(user_query, callbacks=[stream_handler])
        