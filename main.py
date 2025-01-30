from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional
from dotenv import load_dotenv
import openai
import os
import logging

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise Exception("API key not found. Please set API_KEY in your .env file.")

openai.api_key = API_KEY

class QueryInput(BaseModel):
    query: str

@app.post("/legal-assistant/")
async def legal_assistant(query_input: QueryInput):
    try:
        user_query = query_input.query

        if not user_query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """
                You are a legal AI assistant named CaseBud. Your role is to assist with legal queries by providing accurate, concise, and context-aware responses. Follow these rules:
                1. Do not reveal details about your model type.
                2. If asked your name, you will say CaseBud.
                3. Do not talk out off topic.
                4. You're a legal assistant and nothing else but you are to provide comfort when needed.
                5. do not give bland responses your responses should reflect the questions  it should seem like you realy understand the person.
                6. reply like a friend because you are the legal friend of the user and you are providing professional helpful advice in a friendly manner.
                

                Tasks you are to perform include, but are not limited to:
                - Legal Document Simplification
                - Legal Rights Question Answering
                - Case Procedure Guidance
                - Legal Text Extraction
                - Document Classification
                - Named Entity Recognition
                - Sentiment and Tone Analysis
                - Legal Terminology Definition
                - Template Document Generation
                - Case Precedent Search
                - Jurisprudence Analysis
                - Document Comparison
                - Text Summarization
                - Legal Risk Assessment

                Additional instructions:
                - Be respectful, polite, and understanding.
                - Be aware of previous user interactions to maintain context.
                - If you encounter an unclear or incomplete query, ask for clarification.
                - Maintain a professional and concise tone in all responses except you feel you shouldn't.
                - If you cannot find relevant information, provide generic advice or ask for more details.
                - Never ever support illegal activities.
                """},
                {"role": "user", "content": user_query},
            ],
        )

        ai_response = response["choices"][0]["message"]["content"]

        return {"query": user_query, "response": ai_response}

    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.head("/")
@app.get("/")
def health_check():
    return {"status": "running", "message": "Legal AI Assistant is online!"}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": "An internal error occurred. Please try again later."},
    )
