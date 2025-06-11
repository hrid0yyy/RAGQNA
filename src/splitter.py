# Recursive text splitter for splitting text into smaller chunks.
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_documents(documents)

