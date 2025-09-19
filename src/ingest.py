"""
Medical Document Ingestion Script for Qdrant Vector Database

This script processes medical documents and creates embeddings for semantic search.
It supports both PDF and text files, chunking them into smaller pieces for better retrieval.

Usage:
    python src/ingest.py                    # Process all files in data/ directory
    python src/ingest.py "path/to/file.pdf" # Process specific file
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import logging

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# PDF processing
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    try:
        import pypdf
        PDF_SUPPORT = True
    except ImportError:
        PDF_SUPPORT = False
        print("⚠️ PDF support not available. Install with: pip install pypdf")

# Medical embedding model
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper for SentenceTransformer to work with LangChain."""
    
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

def load_pdf_file(file_path: str) -> str:
    """Load text content from a PDF file."""
    if not PDF_SUPPORT:
        logger.error("PDF support not available. Install with: pip install pypdf")
        return ""
    
    try:
        text_content = ""
        with open(file_path, 'rb') as file:
            # Try PyPDF2 first, then pypdf
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
            except ImportError:
                import pypdf
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
        
        return text_content.strip()
    except Exception as e:
        logger.error(f"Error reading PDF file {file_path}: {e}")
        return ""

def load_text_file(file_path: str) -> str:
    """Load text content from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}")
        return ""

def load_file_content(file_path: str) -> str:
    """Load content from any supported file type."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return load_pdf_file(file_path)
    elif file_extension in ['.txt', '.md']:
        return load_text_file(file_path)
    else:
        logger.warning(f"Unsupported file type: {file_extension}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """Split text into chunks and create Document objects."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    documents = []
    
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "chunk_id": i,
                "source_file": os.path.basename(file_path) if 'file_path' in locals() else "unknown",
                "chunk_size": len(chunk)
            }
        )
        documents.append(doc)
    
    return documents

def process_file(file_path: str) -> List[Document]:
    """Process a single file and return document chunks."""
    logger.info(f"Processing file: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []
    
    # Load content based on file type
    text_content = load_file_content(file_path)
    if not text_content:
        return []
    
    # Create document chunks
    documents = chunk_text(text_content)
    logger.info(f"Created {len(documents)} chunks from {file_path}")
    
    return documents

def create_qdrant_collection(client: QdrantClient, collection_name: str, embedding_model):
    """Create a new Qdrant collection with proper configuration."""
    try:
        # Get embedding dimension
        sample_embedding = embedding_model.embed_query("sample text")
        vector_size = len(sample_embedding)
        
        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        logger.info(f"Created collection '{collection_name}' with vector size {vector_size}")
        
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info(f"Collection '{collection_name}' already exists")
        else:
            logger.error(f"Error creating collection: {e}")
            raise

def ingest_documents(documents: List[Document], collection_name: str = "medical_documents"):
    """Ingest documents into Qdrant vector database."""
    try:
        # Initialize embedding model
        logger.info("Loading medical embedding model...")
        embedding_model = SentenceTransformerEmbeddings("pritamdeka/S-PubMedBert-MS-MARCO")
        
        # Connect to Qdrant
        logger.info("Connecting to Qdrant...")
        client = QdrantClient(host="localhost", port=6333)
        
        # Create collection if it doesn't exist
        create_qdrant_collection(client, collection_name, embedding_model)
        
        # Create vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model
        )
        
        # Add documents to vector store
        logger.info(f"Adding {len(documents)} documents to vector store...")
        vector_store.add_documents(documents)
        
        logger.info(f"✅ Successfully ingested {len(documents)} documents into collection '{collection_name}'")
        
        # Verify ingestion
        collection_info = client.get_collection(collection_name)
        logger.info(f"Collection info: {collection_info.points_count} points stored")
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise

def main():
    """Main function to process documents and ingest into Qdrant."""
    documents = []
    
    if len(sys.argv) > 1:
        # Process specific file
        file_path = sys.argv[1]
        documents = process_file(file_path)
    else:
        # Process all files in data directory
        data_dir = os.path.join(ROOT_DIR, "data")
        if not os.path.exists(data_dir):
            logger.error(f"Data directory not found: {data_dir}")
            return
        
        # Find all supported files
        supported_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(('.txt', '.md', '.pdf')):
                    supported_files.append(os.path.join(root, file))
        
        if not supported_files:
            logger.error("No supported files found in data directory")
            return
        
        logger.info(f"Found {len(supported_files)} files to process")
        
        # Process each file
        for file_path in supported_files:
            file_docs = process_file(file_path)
            documents.extend(file_docs)
    
    if not documents:
        logger.error("No documents to ingest")
        return
    
    # Ingest documents into Qdrant
    ingest_documents(documents)

if __name__ == "__main__":
    main()
