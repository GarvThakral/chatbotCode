from google import genai
from dotenv import load_dotenv
import os
load_dotenv()

client = genai.Client(api_key = os.getenv("geminiKey"))

def get_answer(top_texts,question):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=(
        "You are a chatbot for a company website. The company has provided data from a document "
        "which has been converted into chunks. You are given the top 3 relevant text chunks based on a user's question. "
        "Using these chunks, answer the question clearly and helpfully.\n\n"
        f"Chunks:\n{top_texts[0]}\n\n{top_texts[1]}\n\n{top_texts[2]}\n\n"
        f"Question: {question}"
        )
    )
    return response.text


