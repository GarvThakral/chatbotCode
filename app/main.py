from fastapi import FastAPI
from app.api.index import router
from fastapi.middleware.cors import CORSMiddleware
import nltk
import os
from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Download NLTK data
    try:
        print("Downloading NLTK data...")
        # Set NLTK data path to a writable directory
        nltk_data_dir = '/tmp/nltk_data'
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        nltk.data.path.append(nltk_data_dir)
        
        # Download required NLTK data
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)  # You might need this too
        print("NLTK data downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
    
    # App is starting up
    yield
    
    # Shutdown: cleanup if needed
    print("Application shutting down...")

app = FastAPI(lifespan=lifespan)
def add_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Replace with ["http://localhost:3000"] in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

add_cors(app)

app.include_router(router)
