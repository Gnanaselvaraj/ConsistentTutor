import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class ConsistentTutorRAG:
    def __init__(self, pdf_dir="data/pdfs", db_dir="vector_db"):
        self.pdf_dir = pdf_dir
        self.db_dir = db_dir

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"}
        )

        self.llm = Ollama(
            model="llama3",
            temperature=0.2
        )

    def ingest_pdfs(self):
        documents = []

        for file in os.listdir(self.pdf_dir):
            if file.endswith(".pdf"):
                loader = PyMuPDFLoader(os.path.join(self.pdf_dir, file))
                documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(documents)

        vectordb = FAISS.from_documents(chunks, self.embeddings)
        vectordb.save_local(self.db_dir)

        return len(chunks)

    def load_chain(self):
        vectordb = FAISS.load_local(
            self.db_dir,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        retriever = vectordb.as_retriever(search_kwargs={"k": 5})

        prompt = PromptTemplate.from_template("""
You are ConsistentTutor.

You MUST answer strictly using the provided context.
If the answer is not found in the context, reply exactly:

"I cannot find this in the provided syllabus materials."

Context:
{context}

Question:
{question}
""")

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain
