# 🏥 Mediguide - Medical Document Search System

A powerful medical document search and retrieval system with AI-powered question answering capabilities using Retrieval-Augmented Generation (RAG).

## 🌟 Features

- **📄 Multi-format Support**: Process both PDF and DOCX medical documents
- **🧠 Semantic Search**: Uses medical-specific embeddings (`pritamdeka/S-PubMedBert-MS-MARCO`)
- **🤖 AI Assistant**: RAG-powered responses using Ollama with LlamaMedicine model
- **🔍 Dual Search Modes**: 
  - Basic search with similarity scores
  - AI-enhanced responses with synthesized answers
- **💾 Efficient Storage**: FAISS vector database for fast retrieval
- **🖥️ Interactive Interface**: Command-line interface with multiple usage modes

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Ollama (for AI Assistant mode)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Mediguide
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv med
   # Windows
   med\Scripts\activate
   # Linux/Mac
   source med/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Ollama (for AI Assistant)**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull Elixpo/LlamaMedicine
   ```

### Usage

#### 1. Ingest Documents
Place your medical documents in the `data/` directory and run:

```bash
# Process all files in data/ directory
python src/ingest.py

# Process specific file
python src/ingest.py "data/your-medical-document.pdf"
```

#### 2. Query Documents

**Interactive Mode (Recommended)**
```bash
python src/query.py
```
- Type questions for basic search
- Type `ai <question>` for AI-assisted responses
- Type `mode` to switch between modes
- Type `quit` to exit

**Command Line Mode**

Basic Search:
```bash
python src/query.py "What are the symptoms of diabetes?"
```

AI Assistant (RAG):
```bash
python src/query.py --rag "What are the main sources of clinical practice guidelines?"
python src/query.py --ai "How to treat hypertension?"
```

## 📁 Project Structure

```
Mediguide/
├── src/
│   ├── ingest.py          # Document ingestion and embedding creation
│   └── query.py           # Search and RAG query system
├── data/                  # Place your medical documents here
├── embeddings/            # Generated FAISS vector databases
├── med/                   # Virtual environment
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Configuration

### Embedding Model
The system uses `pritamdeka/S-PubMedBert-MS-MARCO` for medical-specific embeddings. This model is optimized for biomedical text understanding.

### LLM Model
For AI Assistant mode, the system uses `Elixpo/LlamaMedicine` via Ollama, which is specialized for medical question answering.

### Customization
- **Chunk Size**: Modify `chunk_size=500` in `ingest.py` for different text chunk sizes
- **Search Results**: Adjust `k` parameter in queries for more/fewer results
- **LLM Model**: Change the model in `rag_query()` function for different Ollama models

## 📋 Examples

### Example Documents
The system works with various medical document types:
- Clinical practice guidelines
- Medical research papers
- Treatment protocols
- Drug information sheets
- Patient education materials

### Sample Queries

**Basic Search Examples:**
```bash
python src/query.py "diabetes management protocols"
python src/query.py "side effects of statins"
python src/query.py "hypertension treatment guidelines"
```

**AI Assistant Examples:**
```bash
python src/query.py --rag "What are the latest recommendations for diabetes treatment?"
python src/query.py --ai "Explain the mechanism of action for ACE inhibitors"
python src/query.py --rag "What factors should be considered when choosing antihypertensive therapy?"
```

### Expected Output

**Basic Search:**
```
🎯 Found 3 relevant results for: 'diabetes treatment'

📄 Result 1 (Similarity Score: 0.8245)
--------------------------------------------------
Content: [Relevant document chunk about diabetes treatment...]
Source: [Document metadata]
--------------------------------------------------
```

**AI Assistant (RAG):**
```
🤖 AI Assistant Response:
======================================================================
The main sources of clinical practice guidelines include the American 
Diabetes Association (ADA), the American Association of Clinical 
Endocrinology (AACE), and Diabetes Canada. These organizations provide 
evidence-based recommendations for diagnosis, treatment, and management 
of diabetes, incorporating the latest research findings and clinical trials.
======================================================================

📚 Sources: 3 documents retrieved
```

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure virtual environment is activated
   med\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Ollama Connection Issues**
   ```bash
   # Ensure Ollama is running
   ollama serve
   
   # Verify model is installed
   ollama list
   ollama pull Elixpo/LlamaMedicine
   ```

3. **No Documents Found**
   ```bash
   # Run ingestion first
   python src/ingest.py
   
   # Check if embeddings directory exists
   ls embeddings/
   ```

4. **FAISS Database Issues**
   ```bash
   # Re-run ingestion to recreate database
   python src/ingest.py
   ```

## 🔬 Technical Details

### Architecture
1. **Document Ingestion**: Documents are loaded, chunked, and embedded using medical-specific embeddings
2. **Vector Storage**: FAISS is used for efficient similarity search
3. **Retrieval**: Semantic search finds the most relevant document chunks
4. **Generation**: LLM synthesizes coherent answers from retrieved context

### Performance
- **Embedding Model**: 768-dimensional vectors optimized for medical text
- **Search Speed**: Sub-second search across thousands of document chunks
- **Memory Usage**: Efficient FAISS indexing for large document collections

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Sentence Transformers**: For the medical embedding model
- **LangChain**: For document processing and retrieval components
- **FAISS**: For efficient vector similarity search
- **Ollama**: For local LLM inference
- **LlamaMedicine**: For medical-specific language understanding

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Open an issue on GitHub
3. Review the documentation for configuration options

---

**Note**: This system is designed for research and educational purposes. Always consult with qualified healthcare professionals for medical advice and treatment decisions.
