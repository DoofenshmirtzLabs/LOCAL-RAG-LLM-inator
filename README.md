# LOCAL-RAG-LLM
##created an local rag +llm using embedding functions,vector database and openai api and also used streamlit for front end
this project is a document processing application built with Python, leveraging PyPDF2, python-docx, and python-pptx for handling PDF, Word, and PowerPoint files. It uses SentenceTransformer for generating text embeddings and stores them in custom JSON databases. The system supports querying stored documents by similarity using cosine similarity. Streamlit is employed to provide a user-friendly web interface for document uploads, querying, and displaying results. The application includes user authentication with password hashing and secure API interactions with OpenAI's GPT model for advanced queries.
##how to intialize?
copy the files from the repositry and load them up in vscode
2.set open ai api key or hugging face api(make sure u have access to gemma-2b model)
##thats it open terminal and run commands pip install -r requirmetns.txt and enter command streamlit run v3.py
