# RAG QNA from Documents

A Retrieval-Augmented Generation (RAG) based Question and Answer system that allows you to query documents using natural language.

## Installation

### From GitHub Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-qna-from-doc.git
cd rag-qna-from-doc

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .
```

## Environment Setup

Create a `.env` file in the project root:

```bash
MISTRAL_API_KEY=your-mistral-api-key-here
```

## Quick Start

```python
from src import RAGQNA
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize models
llm = ChatMistralAI(model="mistral-large-2407")
embeddings = MistralAIEmbeddings(model="mistral-embed")

# Create RAG instance
rag = RAGQNA(llm, embeddings)

# Add and process documents
rag.add_files(["document.pdf"])
rag.process_files()

# Query documents
answer = rag.query("What is the main topic?")
print(answer)
```

## Features

- 📄 Support for multiple document formats (PDF, TXT, PPTX)
- 🤖 Mistral AI integration for LLM and embeddings
- 🔍 Chroma vector store for efficient document retrieval
- 🎯 Contextual compression for better results
- 💭 Conversational memory for context-aware responses
- 🐍 Easy-to-use Python API
- ⚡ Command-line interface

## Usage Examples

### Basic Example

See `examples/basic_usage.py` for a complete working example.

### Advanced Usage

```python
from src import RAGQNA
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

# Initialize with custom settings
llm = ChatMistralAI(model="mistral-large-2407", temperature=0.7)
embeddings = MistralAIEmbeddings(model="mistral-embed")

rag = RAGQNA(llm, embeddings)

# Add multiple documents
rag.add_files([
    "document1.pdf",
    "document2.txt",
    "presentation.pptx"
])

# Process all files
rag.process_files()

# Query documents (uses conversational memory)
response = rag.query("Summarize the key findings")
print(response)

# Ask follow-up questions
follow_up = rag.query("Can you elaborate on the first point?")
print(follow_up)

# Clear and start fresh
rag.clear()
```

## Command Line Usage

```bash
# Initialize RAG system with documents
ragqna init --documents ./docs/ --model mistral-large-2407

# Note: Query command requires vector store persistence (future feature)
```

## Project Structure

```
rag-qna-from-doc/
├── src/                    # Main package
│   ├── __init__.py
│   ├── rag_qna.py         # Main RAGQNA class
│   ├── loader.py          # Document loaders
│   ├── splitter.py        # Text splitting
│   ├── chroma_vector_store.py  # Vector store
│   ├── retriever.py       # Document retrieval
│   ├── prompt.py          # Prompt templates
│   └── cli.py            # Command line interface
├── examples/              # Usage examples
│   └── basic_usage.py
├── requirements.txt       # Dependencies
├── setup.py              # Package setup
├── README.md
└── .env                  # Environment variables
```

## API Reference

### RAGQNA Class

```python
class RAGQNA:
    def __init__(self, llm, embedding_function, prompt=None)
    def add_files(self, file_paths: list[str])
    def process_files(self)
    def query(self, question: str) -> str
    def clear(self)
    def get_files(self) -> list[str]
    def get_processed_files(self) -> list[str]
    def number_of_files(self) -> int
    def size_chunks(self) -> int
```

#### Methods Description

- **`__init__(llm, embedding_function, prompt=None)`**: Initialize with LLM, embeddings, and optional custom prompt
- **`add_files(file_paths)`**: Add document files to be processed
- **`process_files()`**: Process all added files, create chunks, and build vector store
- **`query(question)`**: Query the processed documents and return an answer
- **`clear()`**: Clear vector store, processed files, and conversation memory
- **`get_files()`**: Get list of all added files
- **`get_processed_files()`**: Get list of files that have been processed
- **`number_of_files()`**: Get count of added files
- **`size_chunks()`**: Get total number of document chunks created

## Requirements

- Python 3.8+
- Mistral AI API key
- Supported document formats: PDF, TXT, PPTX

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License
