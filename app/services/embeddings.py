# importing required classes
from pypdf import PdfReader
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from app.db.database import store_in_db

def read_and_embedd(file, id):
    # Initialize PdfReader with a file-like object
    reader = PdfReader(file)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    chunk = ""         # Current chunk being built
    chunk_list = []    # List of chunks for batch embedding
    
    # Process one page at a time
    for page in reader.pages:
        page_text = page.extract_text()  # Extract text from the current page
        sentences = sent_tokenize(page_text)  # Tokenize into sentences
        
        # Build chunks incrementally
        for sentence in sentences:
            if len(chunk + sentence) <= 500:
                chunk += sentence  # Add sentence to current chunk
            else:
                if chunk:
                    chunk_list.append(chunk)  # Add full chunk to list
                    # Process in batches of 10
                    if len(chunk_list) >= 10:
                        embeddings = model.encode(chunk_list)
                        store_in_db(chunk_list, embeddings, id)
                        chunk_list = []  # Clear the list after storing
                chunk = sentence  # Start new chunk with current sentence
    
    # Handle remaining chunk
    if chunk:
        chunk_list.append(chunk)
    
    # Process any remaining chunks
    if chunk_list:
        embeddings = model.encode(chunk_list)
        store_in_db(chunk_list, embeddings, id)