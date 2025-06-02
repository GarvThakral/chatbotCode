from fastapi import UploadFile, File, Form, APIRouter, HTTPException
import os
import shutil
from dotenv import load_dotenv
from pydantic import BaseModel
from app.db.database import query_from_database, delete_user_embeddings, has_user_embeddings, store_in_db 
from app.services.gemini_response import get_answer
import requests
import json
import jwt

from pypdf import PdfReader
from nltk import sent_tokenize

# Import the model instance directly from the new module
# It will be None initially, but will be populated after app startup (due to lifespan)
from app.core.models import embedding_model_instance 

SECRET_KEY = "s3cret" 
ALGORITHM = "HS256"

load_dotenv() 

router = APIRouter() 

print(os.getenv("thisVar")) 

@router.get("/")
async def root():
    return {"message": "Hello World"}

@router.get("/info")
async def info_endpoint(token:str): 
    return {"message":"This is the info router of the Chatbot, how may I help you?"}    

class AskRequest(BaseModel):
    user_id: str 
    question: str
    api_key: str 

@router.post("/ask")
async def ask(request: AskRequest):
    url = os.getenv("apiRoute") + "user/apiCall"
    data = {'apiKey': request.api_key}
    response = requests.post(
        url, 
        json=data,  
        headers={'Content-Type': 'application/json'}
    )

    user_input = [request.question] 
    
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
    # Check if the model is loaded, it should be by this point due to lifespan
    if embedding_model_instance is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded. Application startup error.")
    
    embeddings = embedding_model_instance.encode(user_input) # Use the imported instance
    
    result = query_from_database(embeddings[0].tolist(), user_id) 
    print("DB result:", result)
    print("Texts from DB:", result.get("texts", []))

    if not result.get("texts"):
        return {"result": "No relevant documents found for your question."}

    gemini_result = get_answer(result["texts"][0], request.question)
    return {"result": gemini_result}

@router.post("/embedd")
async def embedd(file: UploadFile = File(...), token: str = Form(...)):
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Please upload a PDF file!!")

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
        if embedding_model_instance is None:
            raise HTTPException(status_code=500, detail="Embedding model not loaded. Application startup error.")

        read_and_embedd(file.file, user_id, embedding_model_instance) # Pass the imported instance

        return {
            "pdf_name": file.filename,
            "Content-Type": file.content_type,
            "message": "PDF processed and embeddings stored successfully!"
        }
    except HTTPException:
        raise 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during embedding: {str(e)}")

class TokenRequest(BaseModel):
    token: str

@router.post("/deleteEmbeddings")
async def delete_embeddings_endpoint(request: TokenRequest): 
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
async def embeddings_exist_endpoint(request: TokenRequest): 
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


def read_and_embedd(file, user_id: str, model): 
    reader = PdfReader(file)
    
    chunk = ""         
    chunk_list = []    

    for page in reader.pages:
        page_text = page.extract_text()  
        if not page_text: 
            continue
            
        sentences = sent_tokenize(page_text) 
        
        for sentence in sentences:
            if len(chunk) + len(sentence) <= 500: 
                chunk += sentence + " " 
            else:
                if chunk.strip(): 
                    chunk_list.append(chunk.strip()) 
                    
                    if len(chunk_list) >= 10:
                        embeddings = model.encode(chunk_list, convert_to_tensor=True) 
                        store_in_db(chunk_list, embeddings.tolist(), user_id) 
                        chunk_list = []  
                chunk = sentence + " " 
    
    if chunk.strip():
        chunk_list.append(chunk.strip())
    
    if chunk_list:
        embeddings = model.encode(chunk_list, convert_to_tensor=True)
        store_in_db(chunk_list, embeddings.tolist(), user_id)