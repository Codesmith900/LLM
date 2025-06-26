# app/document_processor.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from . import config

def process_pdf(file_path: str) -> list:
    """Loads a PDF and splits it into chunks."""
    logger.info(f"Processing PDF file: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunked_docs = text_splitter.split_documents(documents)
    logger.info(f"PDF split into {len(chunked_docs)} chunks.")
    return chunked_docs