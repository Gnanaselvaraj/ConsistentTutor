# ConsistentTutor - AI Tutor with Adaptive Instruction Intelligence (AII)

An intelligent, adaptive tutoring system powered by **Retrieval-Augmented Generation (RAG)**, **Large Language Models (LLM)**, and **Adaptive Instruction Intelligence (AII)** — transforming from static QA to true AI tutoring.

## 🚀 Version 2.0 - AII Edition

### What's New
✅ **Adaptive Instruction Intelligence (AII)** - PhD-grade architecture for personalized teaching  
✅ **2-Stage Cognitive Pipeline** - Instruction planning + dynamic execution  
✅ **Streamlit UI** - Professional interactive interface  
✅ **Dynamic Teaching Strategies** - Adaptive explanation styles  
✅ **Context-Aware Retrieval** - Intelligent breadth adjustment  

## 🎯 Key Features

### 1. **AII Planner** (Stage 1: Meta Reasoning)
Analyzes student questions to determine:
- **Question Type**: factual, conceptual, algorithmic, comparative, analytical, etc.
- **Explanation Depth**: short, medium, long, very_long
- **Teaching Strategy**: summary, step_by_step, structured, comparative, deep_reasoning
- **Retrieval Breadth (k)**: 3, 5, 8, 12, or 15 context chunks

### 2. **Adaptive RAG Chain** (Stage 2: Instruction Execution)
Executes the plan to:
- Dynamically select retrieval breadth
- Generate adaptive teaching prompts
- Produce personalized explanations
- Maintain hallucination-aware grounding

### 3. **Streamlit Interface**
- 📚 PDF upload and knowledge base management
- 💬 Interactive chat with the tutor
- 📊 AII strategy visualization
- 📖 Built-in documentation
- ⭐ User feedback integration

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│           Student Question                          │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│         AII PLANNER (LLM)                           │
│  - Analyze question type                            │
│  - Determine depth requirement                      │
│  - Select teaching strategy                         │
│  - Calculate retrieval breadth                      │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│    { type, depth, strategy, k }                     │
│    (Instruction Plan)                               │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│    DYNAMIC RAG RETRIEVER                            │
│    - FAISS vector DB with plan.k                    │
│    - Retrieve contextual chunks                     │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│   ADAPTIVE TEACHING PROMPT                          │
│   - Incorporate plan parameters                     │
│   - Set depth and strategy guidelines               │
│   - Structure answer format                         │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│    LLM GENERATION                                   │
│    - Generate context-grounded answer               │
│    - Follow strategy guidelines                     │
│    - Maintain knowledge base fidelity                │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│      GROUNDED EXPLANATION                           │
│      (Delivered to Student)                         │
└─────────────────────────────────────────────────────┘
```

## 📦 Project Structure

```
ConsistentTutor/
├── rag_engine.py              # Core RAG + AII implementation
├── app.py                     # Streamlit web interface
├── setup.ipynb                # Jupyter notebook for setup
├── requirements.txt           # Python dependencies
├── data/
│   └── pdfs/                 # PDF course materials
├── vector_db/                # FAISS vector database (generated)
└── README.md                 # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Ollama (for local LLM inference)
- pip or conda

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/ConsistentTutor.git
cd ConsistentTutor
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install Ollama** (https://ollama.ai):
```bash
# Download and install Ollama, then start the server
ollama serve
```

4. **Pull a model** (in another terminal):
```bash
ollama pull llama2
```

5. **Add your PDFs**:
   - Place your course materials (PDFs) in `data/pdfs/`

### Running the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

## 🧪 Usage Examples

### Example 1: Factual Question
**Question**: "What is a stack?"  
**Expected Output**:
- Type: `factual`
- Depth: `short`
- k: `3`
- Strategy: `summary`
- Answer: Brief definition (2-3 lines)

### Example 2: Conceptual Question
**Question**: "Explain stack data structure"  
**Expected Output**:
- Type: `conceptual`
- Depth: `long`
- k: `8`
- Strategy: `structured`
- Answer: Detailed explanation with examples

### Example 3: Comparative Question
**Question**: "Compare stack and queue"  
**Expected Output**:
- Type: `comparative`
- Depth: `long`
- k: `12`
- Strategy: `comparative`
- Answer: Side-by-side comparison

### Example 4: Algorithmic Question
**Question**: "Explain A* algorithm with example"  
**Expected Output**:
- Type: `algorithmic`
- Depth: `very_long`
- k: `15`
- Strategy: `step_by_step`
- Answer: Full textbook-style explanation with steps

## 🧠 Understanding AII Parameters

### Question Types
| Type | Example | Depth | k |
|------|---------|-------|---|
| factual | "What is X?" | short | 3-5 |
| conceptual | "Explain X" | long | 8-12 |
| algorithmic | "How to implement X?" | very_long | 12-15 |
| comparative | "X vs Y?" | long | 8-12 |
| analytical | "Why is X important?" | very_long | 12-15 |

### Explanation Depths
- **short** (2-3 lines): Quick reference answers
- **medium** (1 paragraph): Balanced explanations
- **long** (2-3 paragraphs): Detailed with key points
- **very_long** (textbook): Comprehensive with connections

### Teaching Strategies
- **summary**: Bullet points
- **step_by_step**: Numbered sequential steps
- **structured**: Headers and organized sections
- **comparative**: Side-by-side contrasts
- **deep_reasoning**: Abstract principles and connections

## 🔧 Configuration

Edit parameters in `rag_engine.py`:

```python
# Chunk size for PDF splitting
chunk_size=900
chunk_overlap=120

# Embeddings model
model_name="BAAI/bge-small-en-v1.5"

# LLM settings
model="llama2"
temperature=0.2
```

## 📊 Research Contributions

This implementation includes:

1. **Adaptive Instruction Intelligence (AII)** - Meta-reasoning for instruction planning
2. **Dynamic Retrieval Planning** - Context breadth adjustment based on question analysis
3. **Hallucination-Aware Grounded Tutoring** - Strict context adherence
4. **Edge-Deployed Multimodal Tutor** - Local deployment without cloud dependencies

**Publishable as**: IEEE/ACM paper on adaptive tutoring systems

## 🚀 Next Steps (Optional Enhancements)

- [ ] Concept graph generation
- [ ] Topic sequencing & learning paths
- [ ] Progress tracking & student profiling
- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Integration with educational platforms
- [ ] Analytics dashboard
- [ ] Student performance metrics

## 🛠 Troubleshooting

### "Connection refused" error
```bash
# Make sure Ollama is running
ollama serve
```

### "Knowledge base not initialized"
1. Upload PDFs via the sidebar
2. Click "Build Knowledge Base"
3. Wait for indexing to complete

### "JSON parsing error"
- The system will fallback to safe defaults
- Check LLM output format

### Slow responses
- Reduce `chunk_size` in `rag_engine.py`
- Use faster embeddings model
- Use GPU-accelerated FAISS (`faiss-gpu`)

## 📚 Dependencies

- **LangChain**: Orchestration framework
- **Ollama**: Local LLM inference
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Embeddings
- **Streamlit**: Web interface
- **PyMuPDF**: PDF processing

## 📄 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 👨‍💻 Authors

**Selva Raj** - AI/ML Engineer

## 🙏 Acknowledgments

- LangChain community
- Ollama team
- FAISS developers
- Streamlit team

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation tab in the app
- Review example questions in this README

---

**Version**: 2.0.0 (AII Edition)  
**Release Date**: January 2026  
**Status**: Production Ready 🚀
