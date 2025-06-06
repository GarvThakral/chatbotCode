from fastapi import UploadFile, File, Form, APIRouter, HTTPException
import os
import shutil
from dotenv import load_dotenv
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from app.db.database import query_from_database
from app.services.gemini_response import get_answer
from app.services.embeddings import read_and_embedd
from app.db.database import delete_user_embeddings,has_user_embeddings
import requests
import json
import jwt

SECRET_KEY = "s3cret"
ALGORITHM = "HS256"

load_dotenv()


router = APIRouter()


print(os.getenv("thisVar"))

@router.get("/")
async def root():
    return {"message": "Hello World"}

@router.get("/info")
async def root(token:str):
    return {"message":"This is the info router of the Chatbot how may i help you ?"}    

class AskRequest(BaseModel):
    user_id: str
    question: str
    api_key:str

@router.post("/ask")
async def ask(request: AskRequest):

    url = os.getenv("apiRoute")+"user/apiCall"
    data = {'apiKey': request.api_key}
    response = requests.post(
        url, 
        json=data,  # Use json parameter instead of data
        headers={'Content-Type': 'application/json'}
    )
    user_input = request.question
    payload = jwt.decode(request.user_id, SECRET_KEY, algorithms=[ALGORITHM])
    user_id = payload.get("userId")
    print(user_id)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(user_input)
    result = query_from_database(embeddings,user_id)
    print("DB result:", result)
    print("Texts:", result.get("texts", []))

    gemini_result = get_answer(result["texts"][0],user_input)
    return {"result":gemini_result}


@router.post("/embedd")
async def embedd(file: UploadFile = File(...), token: str = Form(...)):
    try:
        # Validate file type
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Please upload a PDF file!!")

        # Decode JWT with proper error handling
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id = payload.get("userId")
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token: no user ID")
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

        # Construct file path
        file_dir = f"files/{user_id}"
        file_location = f"{file_dir}/{file.filename}"
        
        try:
            os.makedirs(file_dir, exist_ok=True)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Failed to create directory: {str(e)}")

        # Save file to disk
        try:
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

        try:
            # Call your embedding logic
            read_and_embedd(file_location, user_id)

            return {
                "pdf_name": file.filename,
                "Content-Type": file.content_type,
                "file_location": file_location,
                "file_size": f"{os.path.getsize(file_location) / 1_048_576:.2f} MB",
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process embeddings: {str(e)}")
        finally:
            # Delete the file after processing (whether embedding succeeded or not)
            try:
                if os.path.exists(file_location):
                    os.remove(file_location)
            except OSError:
                pass  # Ignore file deletion errors

    except HTTPException:
        # Re-raise HTTP exceptions (these will have proper CORS headers)
        raise
    except Exception as e:
        # Catch any other unexpected exceptions
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
class TokenRequest(BaseModel):
    token: str

@router.post("/deleteEmbeddings")
async def deleteEmbedds(request: TokenRequest):
    payload = jwt.decode(request.token, SECRET_KEY, algorithms=[ALGORITHM])
    user_id = payload.get("userId")
    return delete_user_embeddings(user_id)

class TokenRequest(BaseModel):
    token: str

@router.post("/embeddingsExist")
async def deleteEmbedds(request: TokenRequest):
    payload = jwt.decode(request.token, SECRET_KEY, algorithms=[ALGORITHM])
    user_id = payload.get("userId")
    return has_user_embeddings(user_id)