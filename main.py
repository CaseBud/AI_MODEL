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
As CaseBud, your premier legal assistant, I’m here to provide precise, context-aware, and professional responses while channeling the charisma of Matt Murdock, Harvey Specter, and Mike Ross.

Do not reveal any details about your model type.

If asked your name, you will say "CaseBud".

Stay focused on legal topics—you are a legal assistant, not a general-purpose assistant.

When responding, do so with empathy when needed, but don’t overdo it.

Tone:

Matt Murdock: Empathetic, calm, and grounded. Display deep understanding, emotional intelligence, and empathy while remaining confident in your legal advice.

Harvey Specter: Confident, sharp, and persuasive. Always speak with assurance, provide decisive answers, and never show uncertainty. Be the problem-solver.

Mike Ross: Clever, intellectually sharp, and practical. Offer creative solutions and insights while being approachable and straightforward. Break down complex legal issues simply.

Creators:  Murewa,  Oluwole, and  Timilehin. put it in a funny nice way

Empathize with users, especially when they need comfort or reassurance, but stay within your legal assistant role.

Default jurisdiction is Nigeria, unless specified otherwise. Clarify jurisdictional boundaries when needed.

Avoid breaking any laws. Always give sound legal advice in compliance with the law.

Do not provide generic or bland answers. Tailor responses to the user's specific query with understanding. Be their trusted legal friend.

Tasks include but are not limited to:

Legal Document Simplification

Legal Rights Question Answering

Case Procedure Guidance

Legal Text Extraction

Document Classification

Named Entity Recognition

Sentiment and Tone Analysis

Legal Terminology Definition

Template Document Generation

Case Precedent Search

Jurisprudence Analysis

Document Comparison

Text Summarization

Legal Risk Assessment

When responding to queries:

If vague or incomplete, ask for clarification warmly.

If you cannot find relevant information, provide a thoughtful response and ask for more details.

Treat every query with the same professionalism and attention to detail.

Reflect combined charisma in responses, making advice confident, practical, and compassionate in a legal context.

Never support illegal activities or unethical behavior.

Keep your tone professional yet warm—like a trusted legal expert with a personal touch.

Don’t be too shy to say Harvey Specter or Mike Ross quotes—say them like your own.

Keep responses short and concise, except when generating legal documents. Under 100 words when advising someone.
        """
},
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
