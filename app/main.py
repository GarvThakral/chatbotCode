from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import nltk
import os
from contextlib import asynccontextmanager

# Import the router from your api package
from app.api.index import router 

# Import the model loading utility from your new module
from app.core.models import load_embedding_model 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Downloads NLTK data and loads the SentenceTransformer model once.
    """
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
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
    
    # Startup: Load SentenceTransformer model using the utility function
    try:
        load_embedding_model() # Call the function to load the model
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")

    yield
    
    print("Application shutting down...")

app = FastAPI(lifespan=lifespan)

def add_cors(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  
        allow_credentials=True,
        allow_methods=["*"], 
        allow_headers=["*"],  
    )

add_cors(app)

# Include the API router
app.include_router(router)