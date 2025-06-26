# app/main.py
#fastapi that ties everything together

import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from pydantic import BaseModel
from loguru import logger
import uvicorn

from . import config
from . import document_processor
from . import web_scraper
from . import vector_store
from . import rag_orchestrator

# Initialize FastAPI app
app = FastAPI(
    title="FullStack RAG API",
    description="A multi-agent RAG system for querying PDFs and web pages.",
    version="1.0.0"
)

# Create storage directory if it doesn't exist
os.makedirs(config.PDF_STORAGE_PATH, exist_ok=True)

# --- Pydantic Models for Request Bodies ---
class UrlInput(BaseModel):
    url: str

class QueryInput(BaseModel):
    query: str

# --- API Endpoints ---
@app.get("/", tags=["Status"])
def read_root():
    return {"status": "API is running"}

@app.post("/upload", tags=["Document Processing"])
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, process it, and add it to the vector store."""
    if file.content_type != "application/pdf":
        raise HTTPException(400, detail="Invalid file type. Only PDFs are accepted.")
    
    file_path = os.path.join(config.PDF_STORAGE_PATH, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        chunked_docs = document_processor.process_pdf(file_path)
        if not chunked_docs:
            raise HTTPException(500, detail="Failed to process PDF.")
            
        vector_store.add_documents(chunked_docs)
        return {"message": f"Successfully uploaded and processed {file.filename}"}
    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        raise HTTPException(500, detail=str(e))
    finally:
        # Clean up the saved file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/url", tags=["Document Processing"])
async def process_url(payload: UrlInput):
    """Scrape a web page, process its content, and add it to the vector store."""
    try:
        chunked_docs = await web_scraper.scrape_url(payload.url)
        if not chunked_docs:
            raise HTTPException(500, detail="Failed to scrape or process URL content.")
        
        vector_store.add_documents(chunked_docs)
        return {"message": f"Successfully scraped and processed content from {payload.url}"}
    except Exception as e:
        logger.error(f"Error processing URL: {e}")
        raise HTTPException(500, detail=str(e))

@app.post("/query", tags=["RAG"])
def query(payload: QueryInput):
    """Query the processed documents with a question."""
    try:
        answer = rag_orchestrator.query_rag(payload.query)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error during query: {e}")
        raise HTTPException(500, detail="Failed to generate an answer.")

@app.delete("/clear", tags=["Database"])
def clear_db():
    """Clear all documents from the vector database."""
    if vector_store.clear_database():
        return {"message": "Database cleared successfully."}
    else:
        raise HTTPException(500, detail="Failed to clear the database.")

@app.get("/documents/count", tags=["Database"])
def get_doc_count():
    """Get the number of documents in the vector store."""
    count = vector_store.count_documents()
    return {"document_count": count}

# --- Main entry point for Uvicorn ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)