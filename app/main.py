from fastapi import FastAPI
from app.api.index import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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
