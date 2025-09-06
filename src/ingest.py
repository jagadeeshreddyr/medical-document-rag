import os
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

def ingest_documents(doc_path, db_path="embeddings/med_faiss"):
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

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Extract texts from documents
    texts = [doc.page_content for doc in docs]

    # Create FAISS database using texts and embedding model
    db = FAISS.from_texts(texts, embedding_model)
    db.save_local(db_path)

    print(f"✅ Ingested {len(docs)} chunks and saved to {db_path}")

if __name__ == "__main__":
    # Use os.path.join with ROOT_DIR to create proper file path
    doc_file = os.path.join(ROOT_DIR, "data", "Medical_Document_Search_Strategy.docx")
    ingest_documents(doc_file)
