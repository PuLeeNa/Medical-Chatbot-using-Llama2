# 🏥 Medical Chatbot using Llama

An intelligent medical chatbot powered by **Llama 2 7B**, **LangChain**, and **Pinecone** vector database. This RAG (Retrieval-Augmented Generation) application provides accurate medical information by retrieving relevant context from medical documents and generating responses using a locally-hosted language model.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-3.1.0-green)
![LangChain](https://img.shields.io/badge/LangChain-1.1.0-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## ✨ Features

- 🤖 **RAG-based Architecture**: Retrieves relevant medical information from documents before generating responses
- 🏠 **Local LLM**: Runs Llama 2 7B model locally for privacy and cost-effectiveness
- 🎯 **Vector Search**: Uses Pinecone for efficient semantic search across medical documents
- 💬 **Interactive UI**: Clean, responsive chat interface with typing indicators
- 🔒 **Privacy-First**: All model inference happens locally, no data sent to external APIs
- ⚡ **Optimized Performance**: Configurable token limits and context windows
- 📚 **Document Support**: Processes PDF medical documents with PyMuPDF

## 🏗️ Architecture

This is a **RAG (Retrieval-Augmented Generation)** chatbot:

1. **Retrieval**: Searches Pinecone vector database for relevant medical document chunks
2. **Augmentation**: Injects retrieved context into the prompt template
3. **Generation**: Llama 2 generates accurate responses based on the context

```
User Query → Embeddings → Pinecone Search → Context Retrieval →
Prompt Template → Llama 2 Model → Response
```

## 🛠️ Tech Stack

### Core Technologies

- **LLM**: Llama 2 7B Chat (Q4_0 quantized GGUF format)
- **Framework**: LangChain 1.1.0
- **Vector DB**: Pinecone
- **Embeddings**: HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **Backend**: Flask 3.1.0
- **Model Inference**: CTransformers 0.2.27

### Key Libraries

- `langchain-community` - Document loaders and LLM integrations
- `langchain-pinecone` - Pinecone vector store integration
- `langchain-huggingface` - HuggingFace embeddings
- `sentence-transformers` - Text embedding models
- `pymupdf` - PDF document processing
- `python-dotenv` - Environment variable management

## 📦 Prerequisites

- Python 3.10 or higher
- Conda (Anaconda/Miniconda)
- 8GB+ RAM (for Llama 2 7B model)
- Pinecone API key ([Get it here](https://www.pinecone.io/))
- Llama 2 7B Chat GGUF model file

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/PuLeeNa/Medical-Chatbot-using-Llama2.git
cd Medical-Chatbot-using-Llama2
```

### Step 2: Create Conda Environment

```bash
conda create -n mchatbot python=3.10 -y
conda activate mchatbot
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Llama 2 Model

Download the Llama 2 7B Chat GGUF model and place it in the `model/` directory:

```bash
# The model file should be: model/llama-2-7b-chat.Q4_0.gguf
```

You can download from:

- [TheBloke's HuggingFace](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)

## ⚙️ Configuration

### Step 1: Set up Environment Variables

Create a `.env` file in the root directory:

```bash
PINECONE_API_KEY=your_pinecone_api_key_here
```

### Step 2: Prepare Medical Documents

1. Place your medical PDF documents in the `data/` directory
2. Run the indexing script to create vector embeddings:

```bash
python store_index.py
```

This will:

- Load PDFs from `data/` folder
- Split documents into chunks
- Generate embeddings using HuggingFace
- Store vectors in Pinecone index

## 🎮 Usage

### Run the Application

```bash
python app.py
```

The Flask server will start at `http://127.0.0.1:5000`

### Access the Chat Interface

Open your browser and navigate to:

```
http://localhost:5000
```

### Example Queries

- "What is diabetes?"
- "What are the symptoms of hypertension?"
- "Explain the side effects of acetaminophen"
- "How to treat common cold?"

## 📁 Project Structure

```
Medical-Chatbot-using-Llama2/
│
├── app.py                      # Flask application entry point
├── store_index.py              # Script to create Pinecone index
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── .env                        # Environment variables (not in repo)
│
├── src/
│   ├── __init__.py
│   ├── helper.py               # Helper functions (PDF loading, embeddings)
│   └── prompt.py               # Prompt templates
│
├── templates/
│   └── chat.html               # Chat UI template
│
├── static/
│   ├── style.css               # UI styling
│   └── logo.png                # Logo image
│
├── model/
│   └── llama-2-7b-chat.Q4_0.gguf  # LLM model file
│
├── data/
│   └── Medical_book.pdf        # Medical documents (PDFs)
│
└── research/
    └── trials.ipynb            # Jupyter notebook for experiments
```

## 🔍 How It Works

### 1. Document Processing (`store_index.py`)

```python
# Load PDFs
documents = load_pdf("data/")

# Split into chunks
text_chunks = text_split(documents)

# Generate embeddings
embeddings = download_hugging_face_embeddings()

# Store in Pinecone
PineconeVectorStore.from_texts(text_chunks, embeddings, index_name="medical-chatbot")
```

### 2. Query Processing (`app.py`)

```python
# User sends query via Flask
query = "What is diabetes?"

# Retrieve relevant context from Pinecone (k=2 chunks)
retriever = docsearch.as_retriever(search_kwargs={'k': 2})

# Generate response using Llama 2 with context
qa = RetrievalQA.from_chain_type(llm, retriever=retriever)
response = qa.invoke({"query": query})
```

### 3. Model Configuration

```python
llm = CTransformers(
    model="model/llama-2-7b-chat.Q4_0.gguf",
    model_type="llama",
    config={
        "max_new_tokens": 1024,      # Maximum response length
        "temperature": 0.8,           # Creativity (0=deterministic, 1=creative)
        "context_length": 2048        # Total token capacity
    }
)
```

## 🐛 Troubleshooting

### Issue: Token Limit Exceeded

```
Number of tokens (974) exceeded maximum context length (512)
```

**Solution**: Increase `context_length` and `max_new_tokens` in `app.py`

### Issue: Slow Responses

**Solutions**:

- Reduce `max_new_tokens` to 512
- Reduce retrieved documents: `search_kwargs={'k': 1}`
- Add multi-threading: `config={"threads": 8}`
- Use GPU if available: `config={"gpu_layers": 35}`

### Issue: Model Not Found

```
RepositoryNotFoundError: 401 Client Error
```

**Solution**: Use absolute path and add `local_files_only=True`:

```python
llm = CTransformers(
    model="C:/path/to/model/llama-2-7b-chat.Q4_0.gguf",
    local_files_only=True
)
```

### Issue: Conda Not Recognized

**Solution**: Initialize conda for PowerShell:

```powershell
C:\ProgramData\Anaconda3\Scripts\conda.exe init powershell
```

Then restart PowerShell.

## 🎯 Performance Optimization

### Speed Improvements

1. **Reduce Context Window**: Lower `context_length` to 1024
2. **Limit Tokens**: Set `max_new_tokens` to 256-512
3. **Fewer Retrievals**: Use `search_kwargs={'k': 1}`
4. **CPU Threading**: Add `"threads": 8` to config
5. **Use Smaller Model**: Consider TinyLlama or Llama 2 3B

### Memory Optimization

- Use quantized models (Q4_0, Q5_0)
- Limit concurrent requests
- Clear cache periodically

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [Meta AI](https://ai.meta.com/) - Llama 2 model
- [LangChain](https://www.langchain.com/) - RAG framework
- [Pinecone](https://www.pinecone.io/) - Vector database
- [HuggingFace](https://huggingface.co/) - Embeddings models

## 📧 Contact

**PuLeeNa** - [@PuLeeNa](https://github.com/PuLeeNa)

Project Link: [https://github.com/PuLeeNa/Medical-Chatbot-using-Llama2](https://github.com/PuLeeNa/Medical-Chatbot-using-Llama2)

---

⭐ **Star this repository if you find it helpful!**
