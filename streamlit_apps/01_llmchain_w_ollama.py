"""
Simple LLM Chain with Ollama
"""

import streamlit as st
from langchain_community.llms.ollama import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


st.set_page_config(page_title="LLMChain - Ollama Example", page_icon="💬")
st.title("💬 Langchain: Simple LLMChain with Ollama")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM and Langchain")

with st.sidebar:


    st.subheader("Prompt Template")
    template = st.text_area(
        "Prompt Template", value="Tell me a joke about {topic}", height=100
    )

    st.subheader("Models and parameters")
    ollama_model = st.text_input(
        "Ollama Model", key="ollama_model", value="mistral"
    )

    ollama_base_url = st.text_input(
        "Ollama Base URL", key="ollama_base_url", value="http://localhost:11434"
    )


# 1. Setup LLM
llm = Ollama(
    model=ollama_model,
    base_url=ollama_base_url
)

# 2. Setup Prompt
prompt = PromptTemplate(template=template, input_variables=["topic"])

# 3. Setup LLM Chain
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Streamlit - User input handler
if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        response = llm_chain.run(user_query)
        st.write(response)
