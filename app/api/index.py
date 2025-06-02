from fastapi import UploadFile, File, Form, APIRouter, HTTPException
import os
import shutil
from dotenv import load_dotenv
from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer # REMOVE THIS LINE - model is loaded globally
from app.db.database import query_from_database, delete_user_embeddings, has_user_embeddings, store_in_db # Ensure store_in_db is imported
from app.services.gemini_response import get_answer
import requests
import json
import jwt

# For PDF processing and sentence tokenization
from pypdf import PdfReader
from nltk import sent_tokenize

# Import the globally loaded model from the main app file
# Assuming your main FastAPI file is named main.py and is at the project root
from main import global_embedding_model 

SECRET_KEY = "s3cret" # In a real app, load this securely from environment variables
ALGORITHM = "HS256"

load_dotenv() # Load environment variables from .env file

router = APIRouter() # Initialize the API router

print(os.getenv("thisVar")) # This might print None if "thisVar" isn't set

@router.get("/")
async def root():
    """Simple root endpoint for health check."""
    return {"message": "Hello World"}

@router.get("/info")
async def info_endpoint(token:str): # Renamed `root` to `info_endpoint` to avoid conflict
    """Provides information about the chatbot service."""
    # The `token` parameter here is unused in the current implementation, consider its purpose.
    return {"message":"This is the info router of the Chatbot, how may I help you?"}    

class AskRequest(BaseModel):
    """Request model for the /ask endpoint."""
    user_id: str # This is expected to be a JWT token string
    question: str
    api_key: str # API key for external service calls

@router.post("/ask")
async def ask(request: AskRequest):
    """
    Handles user questions by querying a database and getting a Gemini response.
    """
    # Call external API for user authentication/billing
    url = os.getenv("apiRoute") + "user/apiCall"
    data = {'apiKey': request.api_key}
    response = requests.post(
        url, 
        json=data,  # Use json parameter for proper JSON body
        headers={'Content-Type': 'application/json'}
    )
    # You might want to add error handling for `response.status_code` here

    user_input = [request.question] # Use the actual user question
    
    # Decode user ID from JWT token
    try:
        payload = jwt.decode(request.user_id, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("userId")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: no user ID found in payload")
        print(f"Decoded user ID: {user_id}")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token decoding error: {e}")

    # Use the globally loaded SentenceTransformer model
    if global_embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded. Application startup error.")
    
    embeddings = global_embedding_model.encode(user_input)
    
    # Query database with user embeddings
    result = query_from_database(embeddings[0].tolist(), user_id) # Ensure embeddings is a list for DB query
    print("DB result:", result)
    print("Texts from DB:", result.get("texts", []))

    if not result.get("texts"):
        return {"result": "No relevant documents found for your question."}

    # Get answer from Gemini model
    # Assuming gemini_response expects the text of the most relevant document and the user's question
    gemini_result = get_answer(result["texts"][0], request.question)
    return {"result": gemini_result}

@router.post("/embedd")
async def embedd(file: UploadFile = File(...), token: str = Form(...)):
    """
    Uploads a PDF file, extracts text, generates embeddings, and stores them in the database.
    """
    try:
        # Validate file type
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Please upload a PDF file!!")

        # Decode JWT to get user_id
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id = payload.get("userId")
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token: no user ID")
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

        # Ensure the model is loaded before attempting to use it
        if global_embedding_model is None:
            raise HTTPException(status_code=500, detail="Embedding model not loaded. Application startup error.")

        # Process the file directly from the UploadFile object's file-like object
        read_and_embedd(file.file, user_id, global_embedding_model)

        return {
            "pdf_name": file.filename,
            "Content-Type": file.content_type,
            "message": "PDF processed and embeddings stored successfully!"
        }
    except HTTPException:
        raise # Re-raise FastAPI HTTPExceptions directly
    except Exception as e:
        # Catch any other unexpected errors and return a 500
        raise HTTPException(status_code=500, detail=f"Unexpected error during embedding: {str(e)}")

class TokenRequest(BaseModel):
    """Request model for endpoints requiring only a token."""
    token: str

@router.post("/deleteEmbeddings")
async def delete_embeddings_endpoint(request: TokenRequest): # Renamed to avoid conflict
    """Deletes all embeddings associated with a user."""
    try:
        payload = jwt.decode(request.token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("userId")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: no user ID")
        
        result = delete_user_embeddings(user_id)
        if result:
            return {"message": "User embeddings deleted successfully!"}
        else:
            raise HTTPException(status_code=404, detail="No embeddings found for this user or deletion failed.")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting embeddings: {str(e)}")

@router.post("/embeddingsExist")
async def embeddings_exist_endpoint(request: TokenRequest): # Renamed to avoid conflict
    """Checks if a user has existing embeddings."""
    try:
        payload = jwt.decode(request.token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("userId")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: no user ID")
        
        exists = has_user_embeddings(user_id)
        return {"exists": exists}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking embeddings: {str(e)}")


def read_and_embedd(file, user_id: str, model): # Add model as a parameter
    """
    Reads text from a PDF file, chunks it into sentences,
    generates embeddings using the provided model, and stores them in the database.
    """
    reader = PdfReader(file)
    # model = SentenceTransformer('all-MiniLM-L6-v2') # REMOVE THIS LINE - use the passed 'model'
    
    chunk = ""         # Current text chunk being built
    chunk_list = []    # List of chunks for batch embedding

    # Process one page at a time to manage memory
    for page in reader.pages:
        page_text = page.extract_text()  # Extract text from the current page
        if not page_text: # Skip empty pages
            continue
            
        sentences = sent_tokenize(page_text) # Tokenize into sentences using NLTK
        
        # Build chunks incrementally, ensuring they don't exceed a certain length (e.g., 500 characters)
        for sentence in sentences:
            # Check if adding the next sentence exceeds the chunk limit
            if len(chunk) + len(sentence) <= 500: # Check total length before adding
                chunk += sentence + " " # Add sentence and a space
            else:
                if chunk.strip(): # Ensure chunk is not just whitespace
                    chunk_list.append(chunk.strip()) # Add the completed chunk to the list
                    
                    # Process chunks in batches (e.g., 10) to reduce memory spikes during embedding
                    if len(chunk_list) >= 10:
                        embeddings = model.encode(chunk_list, convert_to_tensor=True) # Encode chunks
                        # Store in DB, ensuring the embeddings are converted to Python lists if needed by your DB
                        store_in_db(chunk_list, embeddings.tolist(), user_id) 
                        chunk_list = []  # Clear the list after storing
                chunk = sentence + " " # Start a new chunk with the current sentence
    
    # Handle any remaining chunk after the loop finishes
    if chunk.strip():
        chunk_list.append(chunk.strip())
    
    # Process any remaining chunks that didn't form a full batch
    if chunk_list:
        embeddings = model.encode(chunk_list, convert_to_tensor=True)
        store_in_db(chunk_list, embeddings.tolist(), user_id)