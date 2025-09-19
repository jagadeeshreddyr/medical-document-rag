# Mediguide - Medical Document Search System

Mediguide helps you search through medical documents and get intelligent answers to your questions. Think of it as having a smart medical librarian that can instantly find relevant information from thousands of medical papers, guidelines, and drug handbooks.

The system uses advanced AI to understand your questions and find the most relevant information, then provides clear answers with references to the source documents.

## What can it do?

**Document Processing**
You can throw PDFs, text files, and markdown documents at it. The system will read through everything and make it searchable. Currently working with medical papers, drug handbooks, and clinical guidelines.

**Smart Search**
Instead of just matching keywords, it actually understands what you're asking. Ask "What's the best treatment for diabetes?" and it will find relevant information even if those exact words aren't in the documents.

**Intelligent Answers**
When you ask a question, it doesn't just show you search results. It reads through the relevant documents and gives you a proper answer, complete with references so you can verify everything.

**Quality Scoring**
The system is pretty smart about ranking information. It looks at how well the content matches your question, checks for relevant medical terms, and makes sure you get the most useful information first.

**Multiple Ways to Use It**
You can use it through a command line interface, interactive chat mode, or run batch queries. Whatever works best for your workflow.

## Getting Started

**What you'll need**
- Python 3.8 or newer
- Docker (for the database)
- Ollama (for the AI chat features)

**Setup**

First, grab the code:
```bash
git clone <your-repo-url>
cd Mediguide
```

Set up a Python environment (keeps things clean):
```bash
python -m venv med
# Windows
med\Scripts\activate
# Linux/Mac
source med/bin/activate
```

Install everything:
```bash
pip install -r requirements.txt
```

Get the AI model (this one is specialized for medical questions):
```bash
# Install Ollama from https://ollama.ai first
ollama pull Elixpo/LlamaMedicine
```

Start the database (this stores all your document data):
```bash
# Make sure Docker Desktop is running first
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant

# Check it's working
docker ps
curl http://localhost:6333/
```

**How to use it**

**Step 1: Add your documents**
Put your medical documents (PDFs, text files) in the `data/` folder, then run:
```bash
# Process everything in the data folder
python src/ingest.py

# Or process just one file
python src/ingest.py "data/your-medical-document.pdf"
```

This will read all your documents and make them searchable. It takes a few minutes depending on how many documents you have.

**Step 2: Start asking questions**

The easiest way is interactive mode:
```bash
python src/query.py
```

Then just type your questions like:
- "What are the symptoms of diabetes?"
- "ai What's the best treatment for hypertension?" (the 'ai' prefix gives you smart answers)
- Type "quit" when you're done

You can also run single queries from the command line:
```bash
python src/query.py "What are the symptoms of diabetes?"
python src/query.py --rag "What are the main treatment guidelines for diabetes?"
```

## What's in the project

```
Mediguide/
├── src/
│   ├── ingest.py          # Processes documents and makes them searchable
│   └── query.py           # Handles searches and questions
├── data/                  # Put your medical documents here
│   ├── drug_handbook/     # Drug reference materials
│   ├── train/             # Training datasets
│   └── *.txt, *.pdf       # Your medical documents
├── qdrant_storage/        # Database files (created automatically)
├── med/                   # Python virtual environment
└── requirements.txt       # List of required Python packages
```

## Configuration

**The AI Models**
The system uses two main AI components:
- Document understanding: A model called `pritamdeka/S-PubMedBert-MS-MARCO` that's specifically trained on medical texts
- Question answering: `Elixpo/LlamaMedicine` through Ollama, which is designed for medical Q&A

**How it decides what's relevant**
When you ask a question, the system scores each piece of information on several factors:
- How well it matches the meaning of your question (50% weight)
- Whether it contains your keywords (30% weight)  
- Overall quality of the content (10% weight)
- How much medical terminology it contains (10% weight)

Then it classifies everything as high priority (really relevant), medium priority (somewhat relevant), or supporting information (might be useful).

**Tweaking the system**
You can adjust things like:
- How big the text chunks are (currently 500 characters)
- How many search results to return
- Which AI model to use for answers
- The relevance thresholds for different priority levels

Most of these settings are in the source code files if you want to experiment.

## Examples

**What kind of documents work well?**
- Clinical practice guidelines
- Medical research papers
- Treatment protocols  
- Drug information sheets
- Patient education materials

Basically any medical text that you'd want to search through.

**Example questions you can ask:**

Simple searches:
```bash
python src/query.py "diabetes management protocols"
python src/query.py "side effects of statins"
python src/query.py "hypertension treatment guidelines"
```

Smart AI answers:
```bash
python src/query.py --rag "What are the latest recommendations for diabetes treatment?"
python src/query.py --rag "How do ACE inhibitors work?"
python src/query.py --rag "What should I consider when choosing blood pressure medication?"
python src/query.py --rag --style detailed "How should metformin be dosed in kidney disease?"
```

**What you'll get back**

For basic searches, you get a list of relevant text chunks with similarity scores:
```
Found 3 relevant results for: 'diabetes treatment'

Result 1 (Score: 0.82)
Content: [Relevant text about diabetes treatment...]
Source: diabetes_guidelines.pdf
```

For AI questions, you get a proper answer plus the sources:
```
AI Response:
The main diabetes treatment guidelines come from the American Diabetes Association (ADA), 
American Association of Clinical Endocrinology (AACE), and Diabetes Canada. These 
organizations provide evidence-based recommendations for diagnosis, treatment, and 
management, incorporating the latest research findings.

For kidney disease patients, metformin dosing should be adjusted based on creatinine 
clearance levels...

Sources: 3 documents used
- diabetes_guidelines.txt (very relevant)
- drug_handbook.pdf (relevant)  
- clinical_protocols.txt (supporting info)
```

## Managing the Database

The system uses Qdrant to store all the processed document data. Here's how to work with it:

**Starting the database:**
```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

**Common database commands:**
```bash
# Check if it's running
docker ps

# Stop the database
docker stop qdrant

# Start it again
docker start qdrant

# Remove the container (your data stays safe in qdrant_storage/)
docker rm qdrant

# See what's happening in the logs
docker logs qdrant
```

**Accessing the database:**
- Web interface: http://localhost:6333/dashboard
- API: http://localhost:6333
- Your data is stored in the `qdrant_storage/` folder

## When Things Go Wrong

**"Python can't find the modules"**
Make sure your virtual environment is activated:
```bash
med\Scripts\activate  # Windows
source med/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

**"Can't connect to Ollama" or AI answers don't work**
Ollama might not be running:
```bash
ollama serve
# In another terminal, check if the model is installed
ollama list
# If LlamaMedicine isn't there:
ollama pull Elixpo/LlamaMedicine
```

**"No documents found" when searching**
You probably haven't processed your documents yet:
```bash
python src/ingest.py
```

**Database connection problems**
Check if Docker and Qdrant are running:
```bash
docker ps
# If you don't see qdrant running:
docker start qdrant
# If the container doesn't exist:
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

**Everything seems broken**
Try reprocessing your documents:
```bash
python src/ingest.py
```

Still having issues? Check if the database is responding:
```bash
curl http://localhost:6333/
```

## How it works (technical stuff)

**The basic process:**
1. Documents get chopped up into small pieces (about 500 characters each)
2. Each piece gets converted into a mathematical representation (called an embedding) that captures its meaning
3. When you ask a question, your question also gets converted to the same type of representation
4. The system finds the document pieces that are most similar to your question
5. Those pieces get fed to the AI model which writes a coherent answer

**Performance:**
- Simple searches: Usually under 1 second
- AI answers: 2-10 seconds depending on the question complexity
- Processing documents: About 1,000 text chunks per second
- Current database: Over 5,500 document chunks indexed
- Should handle up to 100,000+ chunks without problems
- Storage: About 500MB for the current test dataset
- Can handle about 10 people using it at the same time

The whole thing runs locally on your computer, so your medical documents never leave your machine.

