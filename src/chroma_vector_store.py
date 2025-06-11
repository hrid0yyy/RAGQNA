from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration
PERSIST_DIRECTORY = "vector_store"
COLLECTION_NAME = "sample"


class VectorStore:
    def __init__(self, embedding_function=None):

        if not embedding_function:
            raise ValueError("Embedding function must be provided")
        
        if not os.path.exists(PERSIST_DIRECTORY):
            os.makedirs(PERSIST_DIRECTORY)

        self.vector_store = Chroma(
            embedding_function=embedding_function,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=COLLECTION_NAME
        )

    def add_documents(self, docs):
        self.vector_store.add_documents(docs)
    
    def load(self):
        return self.vector_store
    
    def clear(self):
        if os.path.exists(PERSIST_DIRECTORY):
            os.rmdir(PERSIST_DIRECTORY)
        else:
            raise FileNotFoundError("Vector store does not exist. Nothing to clear.")
        
