from fastapi import FastAPI
from app.api.index import router

app = FastAPI()
app.include_router(router)

