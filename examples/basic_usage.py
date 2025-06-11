"""Interactive RAG QNA usage with clean state tracking via RAGQNA class."""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import RAGQNA
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from dotenv import load_dotenv

def list_supported_files(directory):
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(('.pdf', '.txt', '.pptx')) and f != 'requirements.txt'
    ]

def display_menu():
    print("\n📌 Menu:")
    print("1. 📂 Add Files")
    print("2. 📄 Show Existing Added Files")
    print("3. 🔢 Number of Files Added")
    print("4. 🔍 Number of Chunks")
    print("5. 📁 Show Processed Files")
    print("6. 🗑️ Clear Vector Store")
    print("7. ❓ Ask a Question")
    print("0. 🚪 Quit")

def main():
    try:
        load_dotenv()

        # Initialize models
        llm = ChatMistralAI(model="mistral-large-2407")
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        rag_instance = RAGQNA(llm, embeddings)

        print("📁 Please place your document files in the project root directory")
        print("📄 Supported formats: PDF, TXT, PPTX")

        while True:
            display_menu()
            choice = input("\n👉 Enter your choice: ").strip()

            if choice == '1':
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                available_files = list_supported_files(parent_dir)

                if not available_files:
                    print("❌ No supported files found.")
                else:
                    print(f"📚 Found {len(available_files)} file(s):")
                    for idx, f in enumerate(available_files):
                        print(f"{idx+1}. {os.path.basename(f)}")

                    file_nums = input("\nSelect file numbers to add (comma separated): ")
                    try:
                        selected = [available_files[int(i)-1] for i in file_nums.split(",") if i.strip().isdigit()]
                        rag_instance.add_files(selected)
                        print(f"✅ Added {len(selected)} file(s).")
                    except Exception as e:
                        print(f"❌ Invalid selection. Error: {str(e)}")

            elif choice == '2':
                files = rag_instance.get_files()
                if not files:
                    print("📭 No files added yet.")
                else:
                    print("📄 Files added:")
                    for f in files:
                        print(f"   - {os.path.basename(f)}")

            elif choice == '3':
                print(f"🔢 Number of files added: {rag_instance.number_of_files()}")

            elif choice == '4':
                print(f"🔍 Number of chunked documents: {rag_instance.size_chunks()}")

            elif choice == '5':
                processed = rag_instance.get_processed_files()
                if not processed:
                    print("📭 No files processed yet.")
                else:
                    print("📁 Processed files:")
                    for f in processed:
                        print(f"   - {os.path.basename(f)}")

            elif choice == '6':
                rag_instance.clear()
                print("✅ Vector store, chunks, and processed files cleared!")

            elif choice == '7':
                if rag_instance.number_of_files() == 0:
                    print("⚠️ No files added. Add files before asking questions.")
                    continue
                if not rag_instance.get_processed_files():
                    print("🔄 Processing files...")
                    try:
                        rag_instance.process_files()
                        print("✅ Documents processed successfully!")
                    except Exception as e:
                        print(f"❌ Failed to process: {str(e)}")
                        continue

                while True:
                    question = input("\n❓ Ask your question (or type 'back' to return): ").strip()
                    if question.lower() == 'back':
                        break
                    elif not question:
                        print("⚠️ Please enter a valid question.")
                        continue

                    print("🔍 Searching for relevant information...")
                    try:
                        response = rag_instance.query(question)
                        print(f"\n🤖 Answer: {response}")
                    except Exception as e:
                        print(f"❌ Error during query: {str(e)}")

            elif choice == '0':
                print("👋 Exiting. Goodbye!")
                break

            else:
                print("❌ Invalid choice. Please try again.")

    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("1. Ensure your .env file contains MISTRAL_API_KEY")
        print("2. Ensure supported files are in the project directory")
        print("3. Run `pip install -e .` if any modules are missing")

if __name__ == "__main__":
    main()
