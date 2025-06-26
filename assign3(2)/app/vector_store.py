# app/vector_store.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document as LangchainDocument
from loguru import logger
import shutil

from . import config

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)

def get_vector_store():
    """Initializes and returns the Chroma vector store."""
    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_PATH,
        embedding_function=embeddings,
    )
    return vector_store

def add_documents(documents: list[LangchainDocument]):
    """Adds documents to the vector store."""
    vector_store = get_vector_store()
    logger.info(f"Adding {len(documents)} documents to the vector store...")
    vector_store.add_documents(documents)
    vector_store.persist()
    logger.info("Documents added and persisted successfully.")

def get_retriever(k_results: int = 5):
    """Returns a retriever for the vector store."""
    vector_store = get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k_results})

def clear_database():
    """Clears all documents from the vector store."""
    try:
        shutil.rmtree(config.CHROMA_PATH)
        logger.info("Database cleared successfully.")
        # Re-initialize to create an empty collection
        get_vector_store().persist()
        return True
    except FileNotFoundError:
        logger.warning("Database path not found, nothing to clear.")
        return False
    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        return False

def count_documents():
    """Counts the number of documents in the vector store."""
    vector_store = get_vector_store()
    try:
        # The _collection.count() method gives the number of embeddings
        return vector_store._collection.count()
    except Exception as e:
        logger.error(f"Could not count documents: {e}")
        return 0