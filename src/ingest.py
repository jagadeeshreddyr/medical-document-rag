import os
import glob
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List

# Set root directory using os.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create a wrapper for SentenceTransformer to work with LangChain
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

# Load a medical embedding model
embedding_model = SentenceTransformerEmbeddings("pritamdeka/S-PubMedBert-MS-MARCO")

def load_single_document(doc_path):
    """Load a single document and return the documents."""
    # Get file extension to choose appropriate loader
    _, file_extension = os.path.splitext(doc_path)
    file_extension = file_extension.lower()
    
    # Choose appropriate loader based on file type
    if file_extension == '.pdf':
        loader = PyPDFLoader(doc_path)
    elif file_extension == '.docx':
        loader = Docx2txtLoader(doc_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Supported: .pdf, .docx")
    
    documents = loader.load()
    
    # Add source metadata to each document
    filename = os.path.basename(doc_path)
    for doc in documents:
        doc.metadata['source_file'] = filename
    
    return documents

def ingest_documents(doc_paths, db_path="embeddings/med_faiss"):
    """Ingest multiple documents into a single FAISS database."""
    all_documents = []
    
    # Load all documents
    for doc_path in doc_paths:
        print(f"📄 Loading {os.path.basename(doc_path)}...")
        documents = load_single_document(doc_path)
        all_documents.extend(documents)
        print(f"   Loaded {len(documents)} pages")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(all_documents)

    # Extract texts from documents
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    # Create FAISS database using texts, metadata, and embedding model
    db = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
    db.save_local(db_path)

    print(f"✅ Ingested {len(docs)} chunks from {len(doc_paths)} documents and saved to {db_path}")

def get_all_documents_in_folder(folder_path):
    """Get all supported document files in a folder."""
    supported_extensions = ['.pdf', '.docx']
    document_files = []
    
    for ext in supported_extensions:
        pattern = os.path.join(folder_path, f"*{ext}")
        files = glob.glob(pattern)
        document_files.extend(files)
    
    return sorted(document_files)

if __name__ == "__main__":
    # Get data folder path
    data_folder = os.path.join(ROOT_DIR, "data")
    
    # Get all document files in the data folder
    document_files = get_all_documents_in_folder(data_folder)
    
    if not document_files:
        print("❌ No supported documents found in the data folder")
        print("   Supported formats: .pdf, .docx")
    else:
        print(f"🔍 Found {len(document_files)} document(s) to ingest:")
        for doc_file in document_files:
            print(f"   - {os.path.basename(doc_file)}")
        print()
        
        # Ingest all documents
        ingest_documents(document_files)
