"""
Simple LLM Chain with OpenAI GPT-3.5
"""

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler


st.set_page_config(page_title="LLMChain QA", page_icon="💬")
st.title("💬 Langchain: Simple QA with LLMChain")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM and Langchain")

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="chatbot_api_key", type="password"
    )
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    if not (openai_api_key.startswith("sk-")):
        st.warning("Please enter your OpenAI API key!", icon="⚠")

    st.subheader("Prompt Template")
    template = st.text_area(
        "Prompt Template", value="Tell me a joke about {topic}", height=100
    )

    st.subheader("Models and parameters")
    temperature = st.sidebar.slider(
        "temperature", min_value=0.01, max_value=1.0, value=0.1, step=0.01
    )


if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()


# Streaming handler to stream response into the Streamlit container
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

# 2. Setup Prompt
prompt = PromptTemplate(template=template, input_variables=["topic"])

# 3. Setup LLM Chain
convo_chain = LLMChain(prompt=prompt, llm=llm)

# Streamlit - User input handler
if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        response = convo_chain.run(user_query, callbacks=[stream_handler])
