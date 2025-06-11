from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv


load_dotenv()


def retriever(vectorstore=None, llm=None, search_type="mmr", k=3, lambda_mult=0.5):

    if vectorstore is None:
        raise ValueError("Vector store must be provided")
    if llm is None:
        raise ValueError("Language model must be provided")

    base_retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, "lambda_mult": lambda_mult}
    )
    compressor = LLMChainExtractor.from_llm(llm)
    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor
    )
