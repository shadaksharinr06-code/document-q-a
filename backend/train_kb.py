import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

def train_knowledge_base():
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Created {DOCS_DIR}/ folder. Place your PDF or TXT files inside and run again.")
        return

    # 1. Load all PDF and TXT documents from the folder
    pdf_loader = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    
    documents = pdf_loader.load() + txt_loader.load()
    if not documents:
        print(f"No PDF or TXT files found in {DOCS_DIR}/")
        print("Please place your .pdf or .txt files inside the folder and re-run.")
        return

    print(f"Loaded {len(documents)} pages/documents.")

    # 2. Split text into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} searchable chunks.")

    # 3. Create embeddings and save permanently to ChromaDB
    print("Generating HuggingFace embeddings (BAAI/bge-small-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print(f"Training complete! Knowledge base saved to {DB_DIR}/")

if __name__ == "__main__":
    train_knowledge_base()
