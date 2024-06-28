import streamlit as st
import os
import fitz
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import torch
import openai
from dotenv import load_dotenv
from transformers import GPT2Tokenizer

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Streamlit
st.title("Chat with Documents")
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Function to preprocess PDF files
def pdf_preprocessing(document_path):
    document = fitz.open(document_path)
    text = ""
    for page in document:
        text += page.get_text()
    return text

# Function to count tokens in a string
def count_tokens(text):
    return len(tokenizer.encode(text))

# Function to split text into chunks respecting token limits
def split_text_into_chunks(text, max_tokens_per_chunk):
    chunks = []
    current_chunk = ""
    current_chunk_tokens = 0
    
    for paragraph in text.split("\n"):
        tokens = tokenizer.encode(paragraph, add_special_tokens=False)
        
        if current_chunk_tokens + len(tokens) > max_tokens_per_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""
            current_chunk_tokens = 0
        
        current_chunk += paragraph + "\n"
        current_chunk_tokens += len(tokens)
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# Function to process user query and generate response
def process_query(document_path, query):
    document_name = os.path.basename(document_path)
    document = fitz.open(document_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    
    max_tokens_per_chunk = 2000
    chunks = split_text_into_chunks(text, max_tokens_per_chunk)
    
    # Create Chroma database from documents with metadata
    db = Chroma()
    for idx, chunk in enumerate(chunks):
        metadata = {"document_name": document_name, "chunk_index": idx}
        db.add_text_with_metadata(chunk, embedding_function, metadata)
    
    # Perform similarity search
    results = db.similarity_search(query)
    
    # Concatenate results until we reach about 800-1000 tokens
    relevant_content = ""
    total_tokens = 0
    for result in results:
        result_tokens = count_tokens(result.page_content)
        if total_tokens + result_tokens > max_tokens_per_chunk:
            break
        relevant_content += result.page_content + "\n"
        total_tokens += result_tokens
    
    return relevant_content

# Function to call OpenAI API and generate response
def call_openai_api(prompt):
    openai.api_key = openai_api_key
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Ensure this is the correct model name
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000,
        temperature=0.9,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    
    return response.choices[0].message['content']

# Main Streamlit application
if __name__ == '__main__':
    # Sidebar for file upload
    st.sidebar.header("Upload your documents")
    uploaded_files = st.sidebar.file_uploader("Choose files", accept_multiple_files=True, type=['pdf', 'docx'])
    
    # Process uploaded files
    if uploaded_files:
        document_names = []
        for uploaded_file in uploaded_files:
            document_names.append(uploaded_file.name)
            with st.spinner(f'Processing {uploaded_file.name}...'):
                # Save the uploaded file temporarily
                temp_dir = "temp"
                os.makedirs(temp_dir, exist_ok=True)
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the document (for PDFs)
                if uploaded_file.type == 'application/pdf':
                    st.write("Document processed successfully!")
    
        st.sidebar.subheader("Select a document to chat with")
        selected_doc = st.sidebar.selectbox("Available documents", document_names)
        
        if selected_doc:
            st.write(f"You selected: {selected_doc}")
            query = st.text_input(f"Enter your query for {selected_doc}:")
            if query:
                with st.spinner('Processing query...'):
                    relevant_content = process_query(file_path, query)
                    prompt = f"Based on the following context, answer the question: {relevant_content}\n\nQuestion: {query}\nAnswer:"
                    result = call_openai_api(prompt)
                    st.write(result)
