from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import openai
from pymongo import MongoClient

# Initialize FastAPI app
app = FastAPI()

# MongoDB connection
client = MongoClient(
    "mongodb+srv://casebudai:hUU6xxz1Ogw8kT2D@casebud.eagcu.mongodb.net/case_bud_dev?retryWrites=true&w=majority&appName=CaseBud"
)
db = client["case_bud_dev"]
logs_collection = db["logs"]

# OpenAI API credentials
openai.api_key = "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"

# Pydantic model for user input
class QueryInput(BaseModel):
    query: str
    user_id: Optional[str] = None  # Optional field to handle user-specific data

# Function to log interactions
def log_interaction(query: str, response: str, metadata: Dict):
    log_entry = {
        "query": query,
        "response": response,
        "metadata": metadata,
    }
    logs_collection.insert_one(log_entry)

# Route for AI assistant
@app.post("/legal-assistant/")
async def legal_assistant(query_input: QueryInput):
    try:
        # Fetch user query
        user_query = query_input.query
        user_id = query_input.user_id

        # Generate a response using OpenAI GPT-3.5
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a legal AI assistant."},
                {"role": "user", "content": user_query},
            ],
        )

        # Extract response content
        ai_response = response["choices"][0]["message"]["content"]

        # Save interaction in the database
        metadata = {"user_id": user_id} if user_id else {}
        log_interaction(user_query, ai_response, metadata)

        # Return response to the user
        return {"query": user_query, "response": ai_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "running", "message": "Legal AI Assistant is online!"}