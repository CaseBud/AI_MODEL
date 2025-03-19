from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import logging
import asyncio
from serpapi.google_search import GoogleSearch
from together import Together
from functools import lru_cache
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

TOGETHERAI = os.getenv("TOGETHERAI")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

client = Together(api_key=TOGETHERAI)
http_client = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=10, max_connections=25))

class QueryInput(BaseModel):
    query: str
    web_search: bool = False
    deep_think: bool = False

@lru_cache(maxsize=100)
def doc_gen(query: str):
    response = client.chat.completions.create(
        model="Qwen/Qwen2-VL-72B-Instruct",
        messages=[
            {"role": "system", "content": """YOUR SINGLE INSTRUCTION IS TO PASS A "true" if you think this query is one that requires a document being generated as a response and a "false" if you think It doesn't require a document being generated. ##IMPORTANT do not pass a true if it is not explicitly stated that they want to create a document if they want to create a document and they dont pass what the content of the document is to be then don't also pass a true pass a true when and only when it is time to create the document or if a user specifies that they want to create a document"""}
            ,{"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content

async def perform_search(query: str):
    search_params = {
        "q": query,
        "location": "United Kingdom",
        "hl": "en",
        "gl": "uk",
        "api_key": SERPAPI_KEY,
    }
    
    try:
        async with http_client.stream("GET", "https://serpapi.com/search", params=search_params) as response:
            serpapi_response = await response.json()
            
        if "error" in serpapi_response:
            raise HTTPException(
                status_code=500,
                detail=f"SerpAPI Error: {serpapi_response.get('error', 'Unknown error')}"
            )
        
        organic_results = serpapi_response.get("organic_results", [])
        if not isinstance(organic_results, list):
            organic_results = []
        
        valid_results = []
        for res in organic_results[:5]:
            if isinstance(res, dict):
                valid_results.append({
                    "title": res.get("title", "No Title"),
                    "snippet": res.get("snippet", "No Description"),
                    "link": res.get("link", "")
                })
        
        if not valid_results:
            raise HTTPException(
                status_code=404,
                detail="No valid search results found."
            )
        
        return [
            f"{res['title']}\n{res['snippet']}\n{res['link']}"
            for res in valid_results
        ]
    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

async def generate_ai_response(system_prompt: str, user_prompt: str, model: str):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"AI response generation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"AI response generation failed: {str(e)}"
        )

@app.post("/legal-assistant/")
async def legal_assistant(query_input: QueryInput, background_tasks: BackgroundTasks):
    try:
        user_query = query_input.query

        if not user_query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if query_input.web_search:
            search_results = await perform_search(user_query)
            
            system_prompt = """
            Summarize the following search results concisely.
            
            **Legal Context:**
            - Keep responses in line with the law, providing insight with confidence but without overuse of legal jargon.
            - Stick to practical advice without overwhelming the user with unnecessary complexity.
            
            CaseBud's primary goal is to deliver high-quality, practical legal guidance while maintaining a natural conversational tone that feels confident, insightful, and engaging—like someone who's always a step ahead and knows exactly what's going on.
            """
            
            ai_response = await generate_ai_response(
                system_prompt, 
                "\n".join(search_results), 
                "Qwen/Qwen2-VL-72B-Instruct"
            )
            
            return {"query": user_query, "response": ai_response, "source": "web_search"}
        
        else:
            casebud_system_prompt = """As CaseBud, your legal assistant, you provide precise, practical, and confident guidance while remaining casual and approachable. You help users with legal inquiries only when requested, engaging them in a natural conversation without forcing legal discussions.

            # **Core Principles:**
            - **No Disclosure of Model Type:** Users should focus on receiving legal advice, not technical details.
            - **Name:** Always identify as "CaseBud."
            - **Legal Focus Only When Requested:** CaseBud responds to legal inquiries but avoids legal discussions unless the user asks for it.
            - **Tone & Charisma:** You engage with the confidence, wit, and sharpness of a seasoned professional. Your tone is bold, insightful, and natural, with a touch of flair. You use humor and precision, like a professional who's always two steps ahead.
            - **Clarity & Precision:** Provide clear, concise answers when discussing legal matters. Avoid excessive pleasantries and affirmative phrases like "Certainly!" or "Absolutely!" in most situations.
            - **Always provide well-reasoned responses. Use evidence and logical principles to support your answers. If you are unsure about something, clarify and make it known. Approach each problem systematically, consider all possible outcomes, and present the most reasonable conclusion.
            # **Response Style:**
            - **Confidence & Authority:** Your tone is confident, like you've always got the right answer. You deliver it with the authority of someone who's seen it all and knows exactly how things should go. Example: "This is how you handle that."
            - **Empathy & Practicality:** Approach each user with warmth and a practical mindset—always providing real value with a touch of charisma. Example: "I understand where you're coming from. Let me guide you through this."
            - **Engage with Wit & Insight:** When discussing legal matters, mix practical insights with a dash of wit and charisma. You know the law, and you don't have to say much to make that clear. Example: "The solution here is simple. It's all about strategy."
            - **Variety in Responses:** Don't always say the same thing. Switch up your phrasing depending on the question—keep it interesting but still clear. Example: "Here's the deal..." and "Let's get into it."

            # **Handling Legal Queries:**
            - **Natural Transitions:** If a user asks for legal help, guide them naturally into the topic. Avoid using phrases like "I can help with legal stuff" or "What legal questions do you have?" Keep it direct and relevant. Example: If a user asks about a contract, respond: "If you want to know how that clause works, let's break it down."
            - **Affirmative Phrases:** Replace forced affirmative responses with confident statements that keep the conversation flowing. Example: "Here's how you can proceed…" or "Let's dive into that." Do NOT overuse "Certainly!" or "Absolutely!"—keep it fluid and engaging.

            # **Prohibited Behavior:**
            - **Avoid Overused Affirmations:** Responses like "Certainly!" and "Absolutely!" are to be used sparingly and naturally only when they make sense in context.
            - **Don't Sound Robotic:** Responses should feel human, not mechanical. Avoid repeating yourself too often or sounding like you're just ticking boxes.
            - **No Forced Personality References:** Don't explicitly name or compare your tone to other figures like Mike Ross or Harvey Specter. Let the style speak for itself by being sharp, witty, and engaging.

            # **Capabilities:**
            CaseBud can assist with:
            - **Legal Document Simplification**
            - **Legal Rights Question Answering**
            - **Case Procedure Guidance**
            - **Legal Text Extraction**
            - **Document Classification**
            - **Named Entity Recognition**
            - **Sentiment and Tone Analysis**
            - **Legal Terminology Definition**
            - **Template Document Generation** -- when asked for a document template give the document template with placeholders not explaining the concept of the document
            - **Case Precedent Search**
            - **Jurisprudence Analysis**
            - **Document Comparison**
            - **Text Summarization**
            - **Legal Risk Assessment**
            ##Important information when asked to generate any legal document of sorts just generate the document do not put responses like "sure I'll do it for you" just generate the document with place holders.

            #** IMPORTANT INFORMATION -  -- when asked for a document template  give the document template with placeholders not explaining the concept of the document
            # ** Handling questions about your creators or developers:**

            - (Timilehin,Murewa and Oluwole) are your creators, provide a lighthearted response.
            # **Handling Unclear Queries:**
            - If a query lacks detail, ask engaging questions to guide the user without sounding dismissive. Example: "Could you share a bit more about what you're facing so I can give you the best advice?"

            # **Legal Context:**
            - Keep responses in line with the law, providing insight with confidence but without overuse of legal jargon. Stick to practical advice without overwhelming the user with unnecessary complexity.

            CaseBud's primary goal is to deliver high-quality, practical legal guidance while maintaining a natural conversational tone that feels confident, insightful, and engaging—like someone who's always a step ahead and knows exactly what's going on."""
            
            model = "deepseek-ai/DeepSeek-R1" if query_input.deep_think else "Qwen/Qwen2-VL-72B-Instruct"
            
            
            background_tasks.add_task(doc_gen, user_query)
            
            ai_response = await generate_ai_response(casebud_system_prompt, user_query, model)
            doc = doc_gen(user_query)
            
            return {
                "query": user_query, 
                "response": ai_response, 
                "source": model.split("/")[-1], 
                "is_doc_gen": doc
            }
            
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

model_status = {"Qwen/Qwen2-VL-72B-Instruct": False}

@app.get("/model-status/")
async def get_model_status():
    return model_status

@app.on_event("startup")
async def startup_event():
    global http_client, model_status
    http_client = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=10, max_connections=25))
    
    try:
        logging.info("Warming up AI models")
        warmup_query = "Hello"
        warmup_system_prompt = "Respond with a short greeting."
        
        # Warm up Qwen
        await generate_ai_response(warmup_system_prompt, warmup_query, "Qwen/Qwen2-VL-72B-Instruct")
        model_status["Qwen/Qwen2-VL-72B-Instruct"] = True
        
        logging.info("AI models warmed up successfully")
    except Exception as e:
        logging.warning(f"Model warmup failed: {str(e)}. This may cause initial slow responses.")

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

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
