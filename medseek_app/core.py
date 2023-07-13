import os
from typing import Any, Dict, List
from dotenv import load_dotenv, find_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
import pinecone

load_dotenv(find_dotenv())

INDEX_NAME = 'medical-query'

pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'], 
    environment=os.environ['PINECONE_ENVIRONMENT_REGION']
)

def run_llm(query: str, 
            chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    doc_search = Pinecone.from_existing_index(
        embedding=embeddings, 
        index_name=INDEX_NAME
    )
    chat = ChatOpenAI(
        verbose=True, 
        temperature=0
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, 
        retriever=doc_search.as_retriever(), 
        return_source_documents=True
    )
    return qa({"question": query, 
               "chat_history": chat_history})
