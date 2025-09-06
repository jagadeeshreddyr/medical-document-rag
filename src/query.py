"""
Medical Document Query System with RAG (Retrieval-Augmented Generation)

This script provides two modes:
1. Basic Search: Returns raw document chunks with similarity scores
2. AI Assistant (RAG): Uses LLM to synthesize coherent answers from retrieved context

Requirements for RAG mode:
- Ollama installed: pip install ollama
- Ollama running locally
- LlamaMedicine model: ollama pull Elixpo/LlamaMedicine

Usage Examples:
- Interactive mode: python src/query.py
- Basic search: python src/query.py "symptoms of diabetes"
- AI Assistant: python src/query.py --rag "What are clinical practice guidelines?"
"""

import os
import sys
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama not installed. Install with: pip install ollama")

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

# Load the same medical embedding model used in ingest.py
embedding_model = SentenceTransformerEmbeddings("pritamdeka/S-PubMedBert-MS-MARCO")

def load_database(db_path="embeddings/med_faiss"):
    """Load the FAISS database from disk."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Please run ingest.py first.")
    
    print(f"📂 Loading database from: {db_path}")
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

def search_medical_documents(query: str, db, k: int = 5):
    """
    Search for relevant medical information based on the query.
    
    Args:
        query (str): The search query
        db: FAISS database instance
        k (int): Number of results to return
    
    Returns:
        List of tuples (document, score)
    """
    print(f"🔍 Searching for: '{query}'")
    
    # Perform similarity search with scores
    results = db.similarity_search_with_score(query, k=k)
    
    return results

def rag_query(question: str, db, model: str = "Elixpo/LlamaMedicine", k: int = 3):
    """
    Perform Retrieval-Augmented Generation (RAG) query.
    
    Args:
        question (str): The medical question
        db: FAISS database instance
        model (str): Ollama model to use
        k (int): Number of documents to retrieve for context
    
    Returns:
        tuple: (llm_response, retrieved_docs)
    """
    if not OLLAMA_AVAILABLE:
        raise ImportError("Ollama is not available. Please install with: pip install ollama")
    
    print(f"🧠 RAG Query: '{question}'")
    print(f"📚 Retrieving {k} relevant documents...")
    
    # Step 1: Retrieve relevant documents
    docs = db.similarity_search(question, k=k)
    
    if not docs:
        return "❌ No relevant documents found for your query.", []
    
    # Step 2: Combine documents into context
    context = "\n\n".join([d.page_content for d in docs])
    
    print(f"🤖 Generating response using {model}...")
    
    # Step 3: Generate response using LLM
    try:
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            }]
        )
        
        return response["message"]["content"], docs
        
    except Exception as e:
        error_msg = f"❌ Error with Ollama model '{model}': {str(e)}"
        print(error_msg)
        print("💡 Make sure Ollama is running and the model is installed:")
        print(f"   ollama pull {model}")
        return error_msg, docs

def format_results(results, query):
    """Format and display search results."""
    print(f"\n🎯 Found {len(results)} relevant results for: '{query}'\n")
    print("=" * 80)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n📄 Result {i} (Similarity Score: {score:.4f})")
        print("-" * 50)
        print(f"Content: {doc.page_content}")
        
        # Display metadata if available
        if hasattr(doc, 'metadata') and doc.metadata:
            print(f"Source: {doc.metadata}")
        
        print("-" * 50)
    
    print("\n" + "=" * 80)

def interactive_query():
    """Interactive query interface with support for both basic search and RAG."""
    try:
        # Load the database
        db_path = os.path.join(ROOT_DIR, "embeddings", "med_faiss")
        db = load_database(db_path)
        print("✅ Database loaded successfully!")
        print("\n🏥 Medical Document Search System")
        print("\n📋 Available modes:")
        print("  1. Basic Search (default) - Shows raw document chunks")
        print("  2. AI Assistant (RAG) - Synthesized answers using LLM")
        print("\n💡 Commands:")
        print("  - Type your question for basic search")
        print("  - Type 'ai <question>' for AI-assisted response")
        print("  - Type 'mode' to switch between modes")
        print("  - Type 'quit' or 'exit' to stop")
        print("-" * 70)
        
        use_rag_mode = False
        
        while True:
            mode_indicator = "🤖 AI" if use_rag_mode else "🔍 Search"
            query = input(f"\n{mode_indicator} > ").strip()
            
            if not query:
                print("⚠️ Please enter a query.")
                continue
                
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'mode':
                use_rag_mode = not use_rag_mode
                mode_name = "AI Assistant (RAG)" if use_rag_mode else "Basic Search"
                print(f"🔄 Switched to {mode_name} mode")
                continue
            
            # Check for AI command
            if query.lower().startswith('ai '):
                query = query[3:].strip()
                force_rag = True
            else:
                force_rag = False
            
            try:
                if use_rag_mode or force_rag:
                    # RAG mode
                    if not OLLAMA_AVAILABLE:
                        print("❌ Ollama not available. Falling back to basic search.")
                        results = search_medical_documents(query, db, k=3)
                        if results:
                            format_results(results, query)
                        else:
                            print("❌ No results found for your query.")
                    else:
                        llm_response, docs = rag_query(query, db)
                        print(f"\n🤖 AI Assistant Response:")
                        print("=" * 70)
                        print(llm_response)
                        print("=" * 70)
                        print(f"\n📚 Sources: {len(docs)} documents retrieved")
                else:
                    # Basic search mode
                    results = search_medical_documents(query, db, k=3)
                    if results:
                        format_results(results, query)
                    else:
                        print("❌ No results found for your query.")
                        
            except Exception as e:
                print(f"❌ Error during search: {str(e)}")
                
    except FileNotFoundError as e:
        print(f"❌ {str(e)}")
        print("💡 Please run 'python src/ingest.py' first to create the database.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def single_query(query_text: str, num_results: int = 5, use_rag: bool = False):
    """
    Perform a single query and return results.
    
    Args:
        query_text (str): The search query
        num_results (int): Number of results to return
        use_rag (bool): Whether to use RAG (LLM-enhanced) mode
    """
    try:
        # Load the database
        db_path = os.path.join(ROOT_DIR, "embeddings", "med_faiss")
        db = load_database(db_path)
        
        if use_rag and OLLAMA_AVAILABLE:
            # RAG mode
            llm_response, docs = rag_query(query_text, db, k=num_results)
            print(f"\n🤖 AI Assistant Response:")
            print("=" * 70)
            print(llm_response)
            print("=" * 70)
            print(f"\n📚 Sources: {len(docs)} documents retrieved")
            return llm_response, docs
        else:
            # Basic search mode
            if use_rag and not OLLAMA_AVAILABLE:
                print("❌ Ollama not available. Using basic search instead.")
            
            results = search_medical_documents(query_text, db, k=num_results)
            
            if results:
                format_results(results, query_text)
            else:
                print("❌ No results found for your query.")
                
            return results
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return []

if __name__ == "__main__":
    # Check if query is provided as command line argument
    if len(sys.argv) > 1:
        # Parse command line arguments
        args = sys.argv[1:]
        use_rag = False
        
        # Check for RAG flags
        if "--rag" in args or "--ai" in args:
            use_rag = True
            args = [arg for arg in args if arg not in ["--rag", "--ai"]]
        
        if args:
            # Single query mode
            query_text = " ".join(args)
            num_results = 3 if use_rag else 5  # Use fewer results for RAG
            
            mode_name = "🤖 AI Assistant (RAG)" if use_rag else "🔍 Basic Search"
            print(f"🎯 Single Query Mode - {mode_name}")
            single_query(query_text, num_results, use_rag)
        else:
            print("❌ No query provided after flags.")
            print("💡 Usage examples:")
            print("   python src/query.py 'What are the symptoms of diabetes?'")
            print("   python src/query.py --rag 'What are the main sources of clinical practice guidelines?'")
            print("   python src/query.py --ai 'How to treat hypertension?'")
    else:
        # Interactive mode
        interactive_query()
