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
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
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

# Qdrant configuration (same as ingest.py)
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "medical_documents"

# Available medical models
MEDICAL_MODELS = {
    "llamedicine": "Elixpo/LlamaMedicine:latest",
    "medllama2": "medllama2:7b-q4_K_S", 
    "medllama": "medllama2:7b-q4_K_S",  # Use medllama2 as fallback
    "default": "Elixpo/LlamaMedicine:latest"
}

# Response styles
RESPONSE_STYLES = ["guideline", "detailed"]

def get_model_name(model_key: str) -> str:
    """Get the full model name from a key."""
    return MEDICAL_MODELS.get(model_key.lower(), MEDICAL_MODELS["default"])

def load_database(use_local=True, use_file_storage=False):
    """Load the Qdrant database."""
    try:
        print(f"📂 Loading collection: {COLLECTION_NAME}")
        
        # Create Qdrant client and vector store
        if use_file_storage:
            # File-based persistent storage
            storage_path = os.path.join(ROOT_DIR, "qdrant_storage")
            if not os.path.exists(storage_path):
                raise FileNotFoundError(f"File storage not found at {storage_path}. Please run 'python src/ingest.py --file' first.")
            print(f"💾 Loading from file storage: {storage_path}")
            client = QdrantClient(path=storage_path)
            db = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embedding_model
            )
        elif use_local:
            # Server-based storage
            print(f"🔌 Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            db = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embedding_model
            )
        else:
            # In-memory storage
            print("🧠 Using in-memory Qdrant instance...")
            client = QdrantClient(":memory:")
            db = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embedding_model
            )
        
        return db
        
    except Exception as e:
        if use_file_storage:
            raise ConnectionError(f"❌ Error with file storage: {e}")
        elif use_local:
            raise ConnectionError(
                f"❌ Could not connect to Qdrant server: {e}\n"
                "💡 Make sure Qdrant server is running:\n"
                "   - Docker: docker run -p 6333:6333 qdrant/qdrant\n"
                "   - Or try file storage: python src/query.py --file\n"
                "   - Or try memory: python src/query.py --memory"
            )
        else:
            raise ConnectionError(f"❌ Error with in-memory Qdrant: {e}")

def deduplicate_documents(results, similarity_threshold=0.95):
    """
    Remove duplicate or highly similar documents from search results.
    
    Args:
        results: List of tuples (document, score)
        similarity_threshold: Threshold for considering documents as duplicates
    
    Returns:
        List of tuples (document, score) with duplicates removed
    """
    if not results:
        return results
    
    unique_results = []
    seen_content = set()
    
    for doc, score in results:
        content = doc.page_content.strip()
        
        # Check for exact duplicates first
        if content in seen_content:
            continue
        
        # Check for highly similar content
        is_duplicate = False
        for seen in seen_content:
            # Simple similarity check based on content length and overlap
            if len(content) > 50 and len(seen) > 50:
                # Calculate Jaccard similarity for longer texts
                content_words = set(content.lower().split())
                seen_words = set(seen.lower().split())
                
                if content_words and seen_words:
                    intersection = len(content_words.intersection(seen_words))
                    union = len(content_words.union(seen_words))
                    jaccard_sim = intersection / union if union > 0 else 0
                    
                    if jaccard_sim > similarity_threshold:
                        is_duplicate = True
                        break
            else:
                # For shorter texts, use exact match
                if content == seen:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_results.append((doc, score))
            seen_content.add(content)
    
    return unique_results

def search_medical_documents(query: str, db, k: int = 5, deduplicate: bool = True):
    """
    Search for relevant medical information based on the query.
    
    Args:
        query (str): The search query
        db: Qdrant database instance
        k (int): Number of results to return
        deduplicate (bool): Whether to remove duplicate results
    
    Returns:
        List of tuples (document, score)
    """
    print(f"🔍 Searching for: '{query}'")
    
    # Retrieve more results than needed to account for deduplication
    search_k = k * 2 if deduplicate else k
    
    # Perform similarity search with scores
    results = db.similarity_search_with_score(query, k=search_k)
    
    if deduplicate:
        # Remove duplicates
        results = deduplicate_documents(results)
        # Trim to requested number
        results = results[:k]
        
        if len(results) < k:
            print(f"📝 Note: After deduplication, found {len(results)} unique results")
    
    return results

def analyze_context_relevance(question: str, docs_with_scores: List, embedding_model):
    """
    Analyze and rank context relevance to the question for better context-aware responses.
    
    Args:
        question (str): The user's question
        docs_with_scores (List): List of (document, similarity_score) tuples
        embedding_model: The embedding model for semantic analysis
    
    Returns:
        List: Reranked documents with context relevance scores
    """
    question_embedding = embedding_model.embed_query(question)
    question_words = set(question.lower().split())
    
    enhanced_docs = []
    
    for doc, similarity_score in docs_with_scores:
        content = doc.page_content.lower()
        content_words = set(content.split())
        
        # Calculate keyword overlap
        keyword_overlap = len(question_words.intersection(content_words)) / len(question_words)
        
        # Calculate content quality score
        content_length_score = min(len(doc.page_content) / 1000, 1.0)  # Longer content gets higher score
        
        # Medical term density (approximate)
        medical_terms = ['treatment', 'diagnosis', 'therapy', 'patient', 'clinical', 'study', 'guideline', 'recommendation']
        medical_density = sum(1 for term in medical_terms if term in content) / len(medical_terms)
        
        # Combined relevance score
        context_relevance = (
            similarity_score * 0.5 +           # Original similarity
            keyword_overlap * 0.3 +            # Keyword relevance
            content_length_score * 0.1 +       # Content richness
            medical_density * 0.1              # Medical context
        )
        
        enhanced_docs.append({
            'doc': doc,
            'similarity_score': similarity_score,
            'context_relevance': context_relevance,
            'keyword_overlap': keyword_overlap,
            'medical_density': medical_density
        })
    
    # Sort by context relevance
    enhanced_docs.sort(key=lambda x: x['context_relevance'], reverse=True)
    return enhanced_docs

def build_context_aware_prompt(question: str, enhanced_docs: List, style: str = "guideline"):
    """
    Build a context-aware prompt that prioritizes the most relevant information.
    
    Args:
        question (str): The user's question
        enhanced_docs (List): Documents with relevance scores
        style (str): Response style
    
    Returns:
        str: Enhanced prompt for better context-aware responses
    """
    # Analyze question type for better context handling
    question_lower = question.lower()
    question_type = "general"
    
    if any(word in question_lower for word in ['symptom', 'sign', 'present']):
        question_type = "symptoms"
    elif any(word in question_lower for word in ['treatment', 'therapy', 'manage', 'treat']):
        question_type = "treatment"
    elif any(word in question_lower for word in ['diagnos', 'test', 'criteria']):
        question_type = "diagnosis"
    elif any(word in question_lower for word in ['prevent', 'avoid', 'reduce risk']):
        question_type = "prevention"
    elif any(word in question_lower for word in ['cause', 'etiology', 'pathophysiology']):
        question_type = "causation"
    
    # Build context with relevance indicators
    context_parts = []
    primary_sources = set()
    
    for i, doc_info in enumerate(enhanced_docs, 1):
        doc = doc_info['doc']
        relevance = doc_info['context_relevance']
        similarity = doc_info['similarity_score']
        
        # Extract source information
        source_file = "Unknown source"
        if hasattr(doc, 'metadata') and doc.metadata:
            source_file = doc.metadata.get('source_file', 'sample_diabetes_guideline.txt')
            primary_sources.add(source_file)
        
        # Priority indicator based on relevance
        priority = "HIGH" if relevance > 0.8 else "MEDIUM" if relevance > 0.6 else "SUPPORTING"
        
        context_part = f"""[CONTEXT {i} - {priority} RELEVANCE]
Source: {source_file}
Similarity Score: {similarity:.3f}
Content: {doc.page_content}"""
        
        context_parts.append(context_part)
    
    context = "\n\n".join(context_parts)
    
    # Question-type specific instructions
    type_instructions = {
        "symptoms": "Focus on clinical presentations, signs, and symptoms. Organize by severity or frequency when possible.",
        "treatment": "Prioritize first-line treatments, contraindications, and dosing when available. Include alternative options.",
        "diagnosis": "Emphasize diagnostic criteria, cut-off values, and recommended testing procedures.",
        "prevention": "Highlight preventive measures, risk factor modification, and screening recommendations.",
        "causation": "Explain underlying mechanisms, risk factors, and pathophysiology when mentioned.",
        "general": "Provide comprehensive information relevant to the specific question asked."
    }
    
    if style == "guideline":
        enhanced_prompt = f"""You are an expert medical AI assistant. Analyze the provided medical literature and deliver a context-aware, evidence-based clinical response.

MEDICAL LITERATURE (Ranked by Relevance):
{context}

QUESTION: {question}
QUESTION TYPE: {question_type.upper()}

CONTEXT-AWARE INSTRUCTIONS:
- Pay special attention to HIGH RELEVANCE contexts when formulating your response
- {type_instructions.get(question_type, type_instructions["general"])}
- Synthesize information across all provided contexts for a comprehensive answer
- When multiple sources provide the same information, indicate consensus
- When sources differ, explain the variations and their contexts
- Use specific values, criteria, and recommendations when mentioned
- Reference the most authoritative sources (e.g., ADA, WHO, clinical guidelines)
- Maintain clinical accuracy and professional medical terminology

RESPONSE STRUCTURE:
1. **Primary Recommendation/Answer**: Lead with the most direct answer to the question
2. **Supporting Evidence**: Include relevant details from the literature
3. **Clinical Context**: Mention important considerations, contraindications, or variations
4. **Source Attribution**: Clearly indicate which recommendations come from which sources

RESPONSE:"""

    else:  # detailed style
        enhanced_prompt = f"""You are an expert medical AI assistant providing comprehensive, context-aware clinical analysis based on the provided medical literature.

MEDICAL LITERATURE (Ranked by Relevance):
{context}

QUESTION: {question}
QUESTION TYPE: {question_type.upper()}

CONTEXT-AWARE ANALYSIS INSTRUCTIONS:
- Provide a thorough analysis prioritizing HIGH RELEVANCE contexts
- {type_instructions.get(question_type, type_instructions["general"])}
- Cross-reference information between sources and note agreements/discrepancies
- Include specific numerical values, ranges, and criteria when provided
- Explain the clinical reasoning behind recommendations
- Address different patient populations or scenarios when mentioned
- Include evidence levels, recommendation grades, and study details when available
- Maintain strict adherence to the provided literature - do not extrapolate beyond given information

COMPREHENSIVE RESPONSE STRUCTURE:
1. **Executive Summary**: Concise answer to the main question
2. **Detailed Analysis**: Thorough examination of all relevant literature
3. **Evidence Synthesis**: How different sources align or diverge
4. **Clinical Implications**: Practical applications and considerations
5. **Source Documentation**: Clear attribution of information to specific sources
6. **Limitations**: What questions remain unanswered by the provided literature

Begin your response with: "Based on comprehensive analysis of the provided medical literature..."

RESPONSE:"""
    
    return enhanced_prompt

def rag_query(question: str, db, model: str = "Elixpo/LlamaMedicine", k: int = 5, style: str = "guideline"):
    """
    Perform context-aware Retrieval-Augmented Generation (RAG) query with enhanced relevance analysis.
    
    Args:
        question (str): The medical question
        db: Qdrant database instance
        model (str): Ollama model to use (LlamaMedicine, MedLlama2, etc.)
        k (int): Number of unique documents to retrieve for context
        style (str): Response style - "guideline" for concise clinical guidelines, "detailed" for comprehensive analysis
    
    Returns:
        tuple: (llm_response, retrieved_docs, context_analysis)
    """
    if not OLLAMA_AVAILABLE:
        raise ImportError("Ollama is not available. Please install with: pip install ollama")
    
    print(f"🧠 Context-Aware RAG Query: '{question}'")
    print(f"📚 Retrieving {k} relevant documents...")
    
    # Step 1: Retrieve relevant documents with deduplication
    results = search_medical_documents(question, db, k=k, deduplicate=True)
    
    if not results:
        return "❌ No relevant documents found for your query.", [], {}
    
    print(f"🔍 Analyzing context relevance for {len(results)} documents...")
    
    # Step 2: Enhanced context analysis and ranking
    enhanced_docs = analyze_context_relevance(question, results, embedding_model)
    
    # Step 3: Build context-aware prompt
    enhanced_prompt = build_context_aware_prompt(question, enhanced_docs, style)
    
    print(f"🤖 Generating context-aware {style} response using {model}...")
    
    # Display context analysis
    print(f"📊 Context Analysis:")
    for i, doc_info in enumerate(enhanced_docs[:3], 1):  # Show top 3
        print(f"   {i}. Relevance: {doc_info['context_relevance']:.3f}, Similarity: {doc_info['similarity_score']:.3f}")
    
    sources = set()
    for doc_info in enhanced_docs:
        doc = doc_info['doc']
        if hasattr(doc, 'metadata') and doc.metadata:
            source_file = doc.metadata.get('source_file', 'Unknown source')
            sources.add(source_file)
    
    context_analysis = {
        'total_docs': len(enhanced_docs),
        'high_relevance_docs': len([d for d in enhanced_docs if d['context_relevance'] > 0.8]),
        'avg_relevance': sum(d['context_relevance'] for d in enhanced_docs) / len(enhanced_docs),
        'sources': list(sources)
    }
    
    try:
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": enhanced_prompt
            }]
        )
        
        llm_response = response["message"]["content"]
        
        # Add enhanced source summary with context analysis
        source_list = sorted(list(sources))
        source_summary = f"\n\n📚 Context Analysis Summary:"
        source_summary += f"\n   • Total documents analyzed: {context_analysis['total_docs']}"
        source_summary += f"\n   • High-relevance documents: {context_analysis['high_relevance_docs']}"
        source_summary += f"\n   • Average relevance score: {context_analysis['avg_relevance']:.3f}"
        source_summary += f"\n   • Sources: {', '.join(source_list)}"
        
        docs = [doc_info['doc'] for doc_info in enhanced_docs]
        return llm_response + source_summary, docs, context_analysis
        
    except Exception as e:
        error_msg = f"❌ Error with Ollama model '{model}': {str(e)}"
        print(error_msg)
        print("💡 Make sure Ollama is running and the model is installed:")
        print(f"   ollama pull {model}")
        docs = [doc_info['doc'] for doc_info in enhanced_docs] if 'enhanced_docs' in locals() else []
        return error_msg, docs, context_analysis if 'context_analysis' in locals() else {}

def format_results(results, query):
    """Format and display deduplicated search results."""
    print(f"\n🎯 Found {len(results)} unique relevant results for: '{query}'\n")
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

def interactive_query(use_local=True, use_file_storage=False):
    """Interactive query interface with support for both basic search and RAG."""
    try:
        # Load the database
        db = load_database(use_local=use_local, use_file_storage=use_file_storage)
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
                        model_name = get_model_name("default")
                        llm_response, docs, context_analysis = rag_query(query, db, model=model_name, style="guideline")
                        print(f"\n🤖 AI Assistant Response (guideline style):")
                        print("=" * 70)
                        print(llm_response)
                        print("=" * 70)
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

def single_query(query_text: str, num_results: int = 5, use_rag: bool = False, use_local: bool = True, use_file_storage: bool = False, model_key: str = "default", style: str = "guideline"):
    """
    Perform a single query and return results.
    
    Args:
        query_text (str): The search query
        num_results (int): Number of results to return
        use_rag (bool): Whether to use RAG (LLM-enhanced) mode
        use_local (bool): Whether to use local Qdrant server or in-memory mode
        use_file_storage (bool): Whether to use file-based persistent storage
        model_key (str): Model to use for RAG (llamedicine, medllama2, etc.)
        style (str): Response style (guideline, detailed)
    """
    try:
        # Load the database
        db = load_database(use_local=use_local, use_file_storage=use_file_storage)
        
        if use_rag and OLLAMA_AVAILABLE:
            # RAG mode
            model_name = get_model_name(model_key)
            llm_response, docs, context_analysis = rag_query(query_text, db, model=model_name, k=num_results, style=style)
            print(f"\n🤖 AI Assistant Response ({style} style):")
            print("=" * 70)
            print(llm_response)
            print("=" * 70)
            return llm_response, docs, context_analysis
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
        use_local = True  # Default to local Qdrant server
        use_file_storage = False
        model_key = "default"
        style = "guideline"  # Default to guideline style
        
        # Check for flags
        if "--rag" in args or "--ai" in args:
            use_rag = True
            args = [arg for arg in args if arg not in ["--rag", "--ai"]]
        
        # Check for model selection
        if "--model" in args:
            try:
                model_idx = args.index("--model")
                if model_idx + 1 < len(args):
                    model_key = args[model_idx + 1]
                    args = [arg for i, arg in enumerate(args) if i not in [model_idx, model_idx + 1]]
                    print(f"🤖 Using model: {get_model_name(model_key)}")
            except (ValueError, IndexError):
                pass
        
        # Check for style selection
        if "--style" in args:
            try:
                style_idx = args.index("--style")
                if style_idx + 1 < len(args):
                    requested_style = args[style_idx + 1]
                    if requested_style in RESPONSE_STYLES:
                        style = requested_style
                        args = [arg for i, arg in enumerate(args) if i not in [style_idx, style_idx + 1]]
                        print(f"📝 Using {style} response style")
            except (ValueError, IndexError):
                pass
        
        # Check for results count selection
        num_results = 3 if use_rag else 5  # Default values
        if "--results" in args:
            try:
                results_idx = args.index("--results")
                if results_idx + 1 < len(args):
                    requested_results = int(args[results_idx + 1])
                    if requested_results > 0:
                        num_results = requested_results
                        args = [arg for i, arg in enumerate(args) if i not in [results_idx, results_idx + 1]]
                        print(f"📊 Using {num_results} results")
            except (ValueError, IndexError):
                pass
        
        if "--memory" in args:
            use_local = False
            args = [arg for arg in args if arg != "--memory"]
            print("🧠 Using in-memory Qdrant mode")
        elif "--file" in args:
            use_file_storage = True
            use_local = False
            args = [arg for arg in args if arg != "--file"]
            print("💾 Using file-based persistent storage")
        
        if args:
            # Single query mode
            query_text = " ".join(args)
            
            mode_name = "🤖 AI Assistant (RAG)" if use_rag else "🔍 Basic Search"
            if use_file_storage:
                storage_mode = "file-based"
            elif use_local:
                storage_mode = f"server ({QDRANT_HOST}:{QDRANT_PORT})"
            else:
                storage_mode = "in-memory"
            print(f"🎯 Single Query Mode - {mode_name} | Qdrant: {storage_mode}")
            single_query(query_text, num_results, use_rag, use_local, use_file_storage, model_key, style)
        else:
            print("❌ No query provided after flags.")
            print("💡 Usage examples:")
            print("   python src/query.py 'What are the symptoms of diabetes?'")
            print("   python src/query.py --rag 'What are the main sources of clinical practice guidelines?'")
            print("   python src/query.py --rag --style guideline 'First-line diabetes treatment?'")
            print("   python src/query.py --rag --model medllama2 'Treatment recommendations?'")
            print("   python src/query.py --rag --results 5 'What are diabetes symptoms?'")
            print("   python src/query.py --file --rag --style detailed 'Comprehensive diabetes management?'")
            print("   python src/query.py --memory 'query using in-memory mode'")
            print("\n📋 Available models: llamedicine (default), medllama2, medllama")
            print("📋 Available styles: guideline (default), detailed")
            print("📋 Results count: --results N (default: 3 for RAG, 5 for basic search)")
    else:
        # Check for storage mode flags
        use_local = True
        use_file_storage = False
        
        if "--memory" in sys.argv:
            use_local = False
            print("🧠 Using in-memory Qdrant mode")
        elif "--file" in sys.argv:
            use_file_storage = True
            use_local = False
            print("💾 Using file-based persistent storage")
        
        # Interactive mode
        interactive_query(use_local=use_local, use_file_storage=use_file_storage)
