# ConsistentTutor - AI Tutor with RAG

An intelligent tutoring system powered by Retrieval-Augmented Generation (RAG) that combines LLMs with custom document knowledge bases to provide personalized learning experiences.

## Features

- 📚 **RAG Engine**: Retrieves relevant information from custom PDFs
- 🤖 **AI-Powered Responses**: Uses LLMs to generate accurate, contextual answers
- 📄 **PDF Processing**: Automatically extracts and indexes PDF content
- 🔍 **Vector Database**: FAISS-based vector store for efficient similarity search
- 💬 **Interactive Interface**: Flask-based web application for user interaction

## Project Structure

```
ConsistentTutor/
├── app.py                 # Flask web application
├── rag_engine.py          # Core RAG engine implementation
├── setup.ipynb            # Jupyter notebook for setup and configuration
├── data/
│   └── pdfs/             # Directory for storing PDF documents
├── vector_db/            # FAISS vector database (generated)
├── requirements.txt      # Python dependencies (coming soon)
└── README.md             # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ConsistentTutor.git
cd ConsistentTutor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add your PDF files to the `data/pdfs/` directory

4. Run the setup notebook to build the vector database:
```bash
jupyter notebook setup.ipynb
```

## Usage

### Start the Flask Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Using the RAG Engine Directly

```python
from rag_engine import RAGEngine

# Initialize the RAG engine
rag = RAGEngine()

# Query the system
response = rag.query("Your question here")
print(response)
```

## Components

### RAG Engine (`rag_engine.py`)
- Handles PDF loading and processing
- Manages vector database operations
- Implements retrieval and generation logic
- Integrates with LLM for answer generation

### Flask App (`app.py`)
- Web interface for user interactions
- API endpoints for querying
- Request/response handling

## Configuration

Edit `setup.ipynb` to configure:
- PDF processing parameters
- Vector database settings
- LLM model selection
- Retrieval thresholds

## Requirements

Key dependencies:
- `langchain` - LLM framework
- `faiss-cpu` or `faiss-gpu` - Vector database
- `pdf2image` / `pdfplumber` - PDF processing
- `flask` - Web framework
- `sentence-transformers` - Embeddings
- `openai` - LLM API (or alternative)

See `requirements.txt` for the complete list.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Roadmap

- [ ] Add multiple LLM provider support
- [ ] Implement user authentication
- [ ] Add conversation memory
- [ ] Enhance PDF parsing for complex documents
- [ ] Add web UI improvements
- [ ] Implement caching mechanisms
- [ ] Add analytics and logging

## License

This project is open source and available under the MIT License.

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Version**: 1.0.0  
**Last Updated**: January 31, 2026
