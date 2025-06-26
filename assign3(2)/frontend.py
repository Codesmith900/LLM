# frontend.py

import streamlit as st
import requests
import time

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"  # URL of your FastAPI backend

# --- Helper Functions to Interact with the Backend ---

def get_doc_count():
    """Fetches the current document count from the backend."""
    try:
        response = requests.get(f"{API_URL}/documents/count")
        response.raise_for_status()
        return response.json().get("document_count", 0)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching document count: {e}")
        return 0

def add_pdf(file):
    """Uploads a PDF file to the backend."""
    files = {'file': (file.name, file.getvalue(), 'application/pdf')}
    try:
        response = requests.post(f"{API_URL}/upload", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def add_url(url):
    """Sends a URL to the backend for processing."""
    try:
        response = requests.post(f"{API_URL}/url", json={"url": url})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def clear_db():
    """Clears the vector database."""
    try:
        response = requests.delete(f"{API_URL}/clear")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def send_query(query):
    """Sends a query to the RAG chain and gets an answer."""
    try:
        response = requests.post(f"{API_URL}/query", json={"query": query})
        response.raise_for_status()
        return response.json().get("answer", "No answer found.")
    except requests.exceptions.RequestException as e:
        return f"Error querying: {e}"

# --- Streamlit UI ---

st.set_page_config(page_title="FullStack RAG", layout="wide")

st.title("📄 FullStack RAG: Chat with your Documents")
st.write("Upload PDFs or add web URLs, then ask questions about their content.")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Control Panel")

    # Document Count
    st.metric("Documents in DB", get_doc_count())

    st.subheader("Add Content")
    
    # PDF Uploader
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            result = add_pdf(uploaded_file)
            if "error" in result:
                st.error(f"Failed to upload PDF: {result['error']}")
            else:
                st.success(result.get('message', 'PDF processed!'))
            # Force a rerun to update the doc count
            time.sleep(1) # Give a moment for UI to feel responsive
            st.rerun()

    # URL Input
    url_input = st.text_input("Or add a Web URL")
    if st.button("Process URL"):
        if url_input:
            with st.spinner("Scraping and processing URL..."):
                result = add_url(url_input)
                if "error" in result:
                    st.error(f"Failed to process URL: {result['error']}")
                else:
                    st.success(result.get('message', 'URL processed!'))
                time.sleep(1)
                st.rerun()
        else:
            st.warning("Please enter a URL.")

    st.subheader("Database Management")
    if st.button("Clear All Documents", type="primary"):
        with st.spinner("Clearing database..."):
            result = clear_db()
            if "error" in result:
                st.error(f"Failed to clear DB: {result['error']}")
            else:
                st.success(result.get('message', 'Database cleared!'))
            time.sleep(1)
            st.rerun()


# --- Main Chat Interface ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your documents today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = send_query(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})