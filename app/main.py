from fastapi import FastAPI
from app.api.index import router # Assuming app/api/index.py contains your APIRouter
from fastapi.middleware.cors import CORSMiddleware
import nltk
import os
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer # Import here

# Global variable to store the model
global_embedding_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Download NLTK data
    try:
        print("Downloading NLTK data...")
        nltk_data_dir = '/tmp/nltk_data'
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        nltk.data.path.append(nltk_data_dir)
        
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
        print("NLTK data downloaded successfully")

        # Load SentenceTransformer model once globally
        print("Loading SentenceTransformer model...")
        global global_embedding_model
        global_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("SentenceTransformer model loaded successfully")

    except Exception as e:
        print(f"Error during startup: {e}")
    
    yield
    
    print("Application shutting down...")

app = FastAPI(lifespan=lifespan)

def add_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

add_cors(app)

app.include_router(router)