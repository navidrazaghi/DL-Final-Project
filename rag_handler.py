# rag_handler.py

import os
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# Define the path for saving the FAISS index
FAISS_INDEX_PATH = "faiss_index"

def create_or_load_rag_index(pdf_file_path: str):
    """
    Creates and saves a FAISS index from a PDF document or loads it if it already exists.

    This function performs the following steps:
    1. Checks if a FAISS index already exists at the predefined path.
    2. If not, it loads the PDF document from the given file path.
    3. Splits the document's text into smaller, manageable chunks.
    4. Generates embeddings for each chunk using a sentence transformer model.
    5. Creates a FAISS vector store from the embeddings and text chunks.
    6. Saves the created index to disk for future use.
    7. If an index already exists, it loads it directly from the disk.

    Args:
        pdf_file_path (str): The path to the PDF file to be processed.

    Returns:
        FAISS: A FAISS vector store object ready for similarity searches.
    """
    # Use all-MiniLM-L6-v2 for embedding generation
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(FAISS_INDEX_PATH):
        print("No existing FAISS index found. Creating a new one...")
        
        # 1. Load the document
        loader = PyPDFLoader(file_path=pdf_file_path)
        documents = loader.load()
        
        # 2. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        
        # 3. Create FAISS vector store from documents
        print(f"Creating FAISS index from {len(docs)} document chunks...")
        db = FAISS.from_documents(docs, embedding_function)
        
        # 4. Save the index locally
        db.save_local(FAISS_INDEX_PATH)
        print(f"FAISS index created and saved at: {FAISS_INDEX_PATH}")
    else:
        # Load the existing index
        print(f"Loading existing FAISS index from: {FAISS_INDEX_PATH}")
        db = FAISS.load_local(FAISS_INDEX_PATH, embedding_function, allow_dangerous_deserialization=True)

    return db

def get_relevant_context(query: str, db: FAISS) -> str:
    """
    Searches the FAISS index for the most relevant document chunks for a given query.

    Args:
        query (str): The user's query string.
        db (FAISS): The FAISS vector store object.

    Returns:
        str: A concatenated string of the content from the most relevant documents.
    """
    if db is None:
        return ""
    # Perform a similarity search and retrieve the top 3 most relevant chunks
    retrieved_docs = db.similarity_search(query, k=3)
    
    # Combine the content of the retrieved documents into a single context string
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return context