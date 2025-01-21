from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import openai
from pymongo import MongoClient
from dotenv import load_dotenv
import os

app = FastAPI()

load_dotenv()

mongo_uri = os.getenv("MONGO_URL")
openai.api_key = os.getenv("API_KEY")

client = MongoClient(mongo_uri)

db = client["case_bud_dev"]
logs_collection = db["logs"]

class QueryInput(BaseModel):
    query: str
    user_id: Optional[str] = None 

def log_interaction(query: str, response: str, metadata: Dict):
    log_entry = {
        "query": query,
        "response": response,
        "metadata": metadata,
    }
    logs_collection.insert_one(log_entry)

@app.post("/legal-assistant/")
async def legal_assistant(query_input: QueryInput):
    try:
        user_query = query_input.query
        user_id = query_input.user_id
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a legal AI assistant. your role is to assist with legal queries by providing accurate, conscise and context-aware responses."},
                {"role": "user", "content": user_query},
            ],
        )

        ai_response = response["choices"][0]["message"]["content"]
        metadata = {"user_id": user_id} if user_id else {}
        
        log_interaction(user_query, ai_response, metadata)
        
        return {"query": user_query, "response": ai_response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.head("/")
@app.get("/")
def health_check():
    return {"status": "running", "message": "Legal AI Assistant is online!"}