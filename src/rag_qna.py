import os
from .loader import load_doc
from .splitter import split_docs
from .chroma_vector_store import VectorStore
from .retriever import retriever
from .prompt import get_prompt

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class RAGQNA:
    def __init__(self, llm, embedding_function, prompt=None):
        self.llm = llm
        self.embedding_function = embedding_function
        self.files = []
        self.processed_files = []
        self.chunks = []
        self.vector_store = VectorStore(embedding_function)
        self.prompt = prompt or get_prompt()
        
        self.retriever = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.qa_chain = None

    def add_files(self, file_paths: list[str]):
        for file_path in file_paths:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File {file_path} does not exist")
            if file_path not in self.files:
                self.files.append(file_path)

    def process_files(self):
        if not self.files:
            raise ValueError("No files to process")

        for file in self.files:
            if file not in self.processed_files:
                docs = load_doc(file)
                chunks = split_docs(docs)
                self.chunks.append(chunks)
                self.processed_files.append(file)
                self.vector_store.add_documents(chunks)

        # Build retriever
        vs_retriever = retriever(vectorstore=self.vector_store.load(), llm=self.llm)
        self.retriever = vs_retriever

        # Create conversational chain with memory
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vs_retriever,
            memory=self.memory,
            return_source_documents=False
        )

    def query(self, question: str):
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Process documents first.")
        result = self.qa_chain.invoke({"question": question})
        return result["answer"]

    def clear(self):
        self.processed_files.clear()
        self.chunks.clear()
        self.retriever = None
        self.vector_store.clear()
        self.qa_chain = None
        self.memory.clear()

    def get_files(self):
        return self.files

    def size_chunks(self):
        return sum(len(chunk_group) for chunk_group in self.chunks)

    def number_of_files(self):
        return len(self.files)

    def get_processed_files(self):
        return self.processed_files
