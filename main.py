from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from dotenv import load_dotenv
import openai
import os

load_dotenv()

app = FastAPI()

openai.api_key = os.getenv("API_KEY")

class QueryInput(BaseModel):
    query: str

@app.post("/legal-assistant/")
async def legal_assistant(query_input: QueryInput):
    try:
        user_query = query_input.query
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a legal AI assistant. your role is to assist with legal queries by providing accurate, conscise and context-aware responses."},
                {"role": "user", "content": user_query},
            ],
        )

        ai_response = response["choices"][0]["message"]["content"]
        
        return {"query": user_query, "response": ai_response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.head("/")
@app.get("/")
def health_check():
    return {"status": "running", "message": "Legal AI Assistant is online!"}