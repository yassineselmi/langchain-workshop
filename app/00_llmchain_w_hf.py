import streamlit as st
from langchain_community.llms.huggingface_hub  import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler


st.set_page_config(page_title="LLMChain QA", page_icon="💬")
st.title("💬 Langchain: Simple QA with LLMChain")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM and Langchain")

with st.sidebar:
    huggingfacehub_api_token = st.text_input(
        "Hugging Face API Key", key="chatbot_api_key", type="password"
    )
    if not (huggingfacehub_api_token.startswith("sk-")):
        st.warning("Please enter your Hugging face API key!", icon="⚠")


    st.subheader("Prompt Template")
    template = st.text_area(
        "Prompt Template", value="Tell me a joke about {topic}", height=100
    )

    st.subheader("Models and parameters")
    repo_id = st.text_input(
        "Model / Repo ID", value="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
    )
    temperature = st.sidebar.slider(
        "temperature", min_value=0.01, max_value=1.0, value=0.1, step=0.01
    )


if not huggingfacehub_api_token:
    st.info("Please add your Hugging Face API key to continue.")
    st.stop()


if huggingfacehub_api_token:
    # 1. Setup LLM
    llm = HuggingFaceHub(
        huggingfacehub_api_token=huggingfacehub_api_token,
        repo_id=repo_id,
        model_kwargs={"temperature":temperature, "max_length": 4096}
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
