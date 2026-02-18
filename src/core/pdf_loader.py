"""
pdf_loader.py: PDF/document loading utilities
"""
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader

def load_pdfs(pdf_paths: List[str]) -> list:
    docs = []
    for p in pdf_paths:
        docs.extend(PyMuPDFLoader(p).load())
    return docs
