import streamlit as st
import os
from rag_engine import ConsistentTutorRAG

st.set_page_config(page_title="ConsistentTutor", layout="wide")

st.title("📘 ConsistentTutor — Grounded On-Device AI Tutor")

tutor = ConsistentTutorRAG()

st.sidebar.header("📂 Upload Learning Materials")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF textbooks / notes",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("data/pdfs", exist_ok=True)

    for file in uploaded_files:
        with open(f"data/pdfs/{file.name}", "wb") as f:
            f.write(file.getbuffer())

    if st.sidebar.button("Build Knowledge Base"):
        with st.spinner("Processing documents..."):
            chunks = tutor.ingest_pdfs()
        st.sidebar.success(f"Indexed {chunks} knowledge chunks")

if os.path.exists("vector_db/index.faiss"):
    qa_chain = tutor.load_chain()

    st.subheader("Ask your Tutor")

    query = st.text_input("Enter your question")

    if st.button("Ask"):
        with st.spinner("Generating grounded answer..."):
            answer = qa_chain.invoke(query)

        st.markdown("### Answer")
        st.write(answer)
