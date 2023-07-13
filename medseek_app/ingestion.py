import os
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

load_dotenv(find_dotenv())

tmp_file_path = "data/pmid_15329237_6613923.csv"

pinecone.init(api_key=os.environ['PINECONE_API_KEY'], 
              environment=os.environ['PINECONE_ENVIRONMENT_REGION'])


def ingest_docs() -> None:
    loader = CSVLoader(
        file_path=tmp_file_path,
        encoding="utf-8",
        csv_args={
            'delimiter': ','})
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000,
                                                   chunk_overlap=50,
                                                   separators=["\n\n", "\n", " ", ""])
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    Pinecone.from_documents(documents=documents,
                            embedding=embeddings,
                            index_name='medical-query')
    print("Added to Pinecone Vectorstore vectors.")


if __name__ == '__main__':
    ingest_docs()
