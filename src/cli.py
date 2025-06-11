"""Command-line interface for RAG QNA."""

import argparse
import os
import sys
from .rag_qna import RAGQNA
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="RAG-based QNA system")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize RAG system")
    init_parser.add_argument("--documents", "-d", required=True, help="Path to documents directory")
    init_parser.add_argument("--model", "-m", default="mistral-large-2407", help="Model name")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query documents")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--model", "-m", default="mistral-large-2407", help="Model name")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        llm = ChatMistralAI(model=args.model)
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        
        if args.command == "init":
            rag = RAGQNA(llm, embeddings)
            if os.path.isdir(args.documents):
                files = [os.path.join(args.documents, f) for f in os.listdir(args.documents) 
                        if f.endswith(('.pdf', '.txt', '.pptx'))]
                rag.add_files(files)
            else:
                rag.add_files([args.documents])
            rag.process_files()
            print(f"✅ Initialized RAG system with documents from {args.documents}")
            
        elif args.command == "query":
            # Note: In a real implementation, you'd need to persist/load the vector store
            print("❌ Please initialize the system first with 'ragqna init --documents <path>'")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
