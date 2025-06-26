# app/web_scraper.py

import aiohttp
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from loguru import logger

from . import config

async def scrape_url(url: str) -> list:
    """Fetches content from a URL, cleans it, and splits it into chunks."""
    logger.info(f"Scraping URL: {url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                html_content = await response.text()

        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        clean_text = "\n".join(line for line in lines if line)

        if not clean_text:
            logger.warning(f"No text content found at {url}")
            return []

        # Create a single document for the whole page
        doc = LangchainDocument(page_content=clean_text, metadata={"source": url})
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        chunked_docs = text_splitter.split_documents([doc])
        logger.info(f"URL content split into {len(chunked_docs)} chunks.")
        return chunked_docs

    except Exception as e:
        logger.error(f"Failed to scrape {url}: {e}")
        return []