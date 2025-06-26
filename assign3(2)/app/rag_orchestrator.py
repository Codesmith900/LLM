# app/rag_orchestrator.py

from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from loguru import logger

from . import vector_store
from . import config

def get_rag_chain():
    """Creates and returns the RAG chain."""
    retriever = vector_store.get_retriever()
    llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL)
    
    template = """
    Answer the question based only on the following context.
    If you don't know the answer, just say that you don't know. Do not make up an answer.

    Context:
    {context}

    Question:
    {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def query_rag(query_text: str) -> str:
    """Queries the RAG chain and returns the answer."""
    logger.info(f"Received query: {query_text}")
    rag_chain = get_rag_chain()
    response = rag_chain.invoke(query_text)
    logger.info("Generated response successfully.")
    return response