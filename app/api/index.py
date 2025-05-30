from fastapi import APIRouter
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from app.db.database import query_from_database
from app.services.gemini_response import get_answer
load_dotenv()

router = APIRouter()

print(os.getenv("thisVar"))

@router.get("/")
async def root():
    return {"message": "Hello World"}

@router.get("/info")
async def root():
    return {"message":"This is the info router of the Chatbot how may i help you ?"}    

class AskRequest(BaseModel):
    user_id: int
    question: str

@router.post("/ask")

@router.post("/ask")
async def ask(request: AskRequest):
    user_input = ["What is this document about ?"]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(user_input)
    result = query_from_database(embeddings,21)
    gemini_result = get_answer(result["texts"][0],user_input)
    return {"result":gemini_result}