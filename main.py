from typing import Set

from core import run_llm
import streamlit as st
from streamlit_chat import message

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = [source_urls]
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

# App introduction in the sidebar
st.sidebar.markdown(
    """
    <h1 style='font-size:24px;'>Welcome to MedSeek!</h1>
    <p style='font-size:16px;'>MedSeek is a Tool designed specifically for medical data from any formats! \
        It performs Text Summarization, Question Answering, Entity Extraction for the source data.</p>
    """
    , unsafe_allow_html=True
)

# About section
st.markdown(
    """
    <h2>About</h2>
    <p>MedSeek leverages large language models to extract key insights from biomedical text data stored in Vector Stores.</p>
    """
    , unsafe_allow_html=True
)


st.header("Welcome to MedSeek!")
if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state['chat_answers_history'] = []
    st.session_state['user_prompt_history'] = []
    st.session_state['chat_history'] = []

prompt = st.text_input("Prompt", placeholder="Enter your Query here...") or st.button("Submit")

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt, 
            chat_history=st.session_state['chat_history']
        )
        sources = set(
            [doc.metadata['source'] for doc in generated_response['source_documents']]
        )
        formatted_response = (
            f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        )
        st.session_state.chat_history.append((prompt, generated_response['answer']))
        st.session_state.user_prompt_history.append(prompt)
        st.session_state.chat_answers_history.append(formatted_response)

if st.session_state['chat_answers_history']:
    for generated_response, user_query in zip(
        st.session_state['chat_answers_history'], 
        st.session_state['user_prompt_history'], 
    ):
        message(
            user_query, 
            is_user=True
        )
        message(generated_response)
