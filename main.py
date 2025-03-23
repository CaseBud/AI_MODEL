from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import os
import logging
import asyncio
from serpapi.google_search import GoogleSearch
from together import Together
from functools import lru_cache
import httpx
import time
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables with fallbacks
TOGETHERAI = os.getenv("TOGETHERAI")
if not TOGETHERAI:
    logger.error("TOGETHERAI environment variable not set")
    raise ValueError("TOGETHERAI environment variable is required")

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
if not SERPAPI_KEY:
    logger.error("SERPAPI_KEY environment variable not set")
    raise ValueError("SERPAPI_KEY environment variable is required")

# Configuration constants
SEARCH_RESULTS_LIMIT = 5
HTTP_TIMEOUT = 30.0  
CACHE_SIZE = 200  
REQUEST_TIMEOUT = 60.0  

# Initialize API client
client = Together(api_key=TOGETHERAI)

# Application state
model_status: Dict[str, bool] = {
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": False,
    "deepseek-ai/DeepSeek-R1": False
}

# Lifespan context manager for app setup and teardown
@asynccontextmanager
async def lifespan(app: FastAPI):
    #  HTTP client with optimized connection settings
    app.state.http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        timeout=httpx.Timeout(timeout=HTTP_TIMEOUT)
    )
    
    # Warm up models
    try:
        logger.info("Warming up AI models")
        warmup_query = "Hello"
        warmup_system_prompt = "Respond with a short greeting."
        
        # Warm up models in parallel
        await asyncio.gather(
            generate_ai_response(warmup_system_prompt, warmup_query, "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", app.state.http_client),
            generate_ai_response(warmup_system_prompt, warmup_query, "deepseek-ai/DeepSeek-R1", app.state.http_client)
        )
        
        model_status["meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"] = True
        model_status["deepseek-ai/DeepSeek-R1"] = True
        logger.info("AI models warmed up successfully")
    except Exception as e:
        logger.warning(f"Model warmup failed: {str(e)}. This may cause initial slow responses.")
    
    yield
    
    # Close HTTP client
    await app.state.http_client.aclose()
    logger.info("HTTP client closed")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Legal AI Assistant API",
    description="API for providing legal assistance using AI models",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class QueryInput(BaseModel):
    query: str = Field(..., min_length=1, description="User query text")
    web_search: bool = Field(False, description="Whether to perform web search")
    deep_think: bool = Field(False, description="Whether to use deep thinking model")

# Response Models
class SearchResult(BaseModel):
    title: str
    snippet: str
    link: str

class ApiResponse(BaseModel):
    query: str
    response: str
    source: str
    is_doc_gen: Optional[str] = None

# Dependency to get HTTP client
async def get_http_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client

# Helper functions
@lru_cache(maxsize=CACHE_SIZE)
def doc_gen(query: str) -> str:
    """
    Determines if a query requires document generation.
    Enhanced with caching to avoid repeated calls.
    """
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": """YOUR SINGLE INSTRUCTION IS TO PASS A "true" if you think this query is one that requires a document being generated as a response and a "false" if you think It doesn't require a document being generated. ##IMPORTANT do not pass a true if it is not explicitly stated that they want to create a document if they want to create a document and they dont pass what the content of the document is to be then don't also pass a true pass a true when and only when it is time to create the document or if a user specifies that they want to create a document"""},
                {"role": "user", "content": query},
            ],
        )
        logger.debug(f"doc_gen execution time: {time.time() - start_time:.2f}s")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in doc_gen: {str(e)}")
        return "false"  # Default to false on error
async def perform_search(query: str, http_client: httpx.AsyncClient) -> List[str]:
    """
    Performs a web search using SerpAPI with improved error handling and timeouts.
    """
    response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": """THIS QUERY IS TO BE SEARCHED THROUGH SERP API I WANT YOU TO REFINE THIS QUERY IN A WAY THAT IT WILL GET THE BEST MEANINGFUL RESULTS #PASS THE REFINED QUERY AND NOTHING ELSE"""},
                {"role": "user", "content": query},
            ],
        )
    refined_query = response.choices[0].message.content
    search_params = {
        "q": refined_query,
        "location": "United Kingdom",
        "hl": "en",
        "gl": "uk",
        "api_key": SERPAPI_KEY,
    }
    
    try:
        start_time = time.time()
        # Replace streaming approach with a direct request
        response = await http_client.get(
            "https://serpapi.com/search", 
            params=search_params,
            timeout=15.0  # Shorter timeout for search
        )
        
        if response.status_code != 200:
            error_text = response.text
            raise HTTPException(
                status_code=response.status_code,
                detail=f"SerpAPI Error: {error_text}"
            )
            
        serpapi_response = response.json()
        
        logger.debug(f"SerpAPI request took {time.time() - start_time:.2f}s")
        
        if "error" in serpapi_response:
            raise HTTPException(
                status_code=500,
                detail=f"SerpAPI Error: {serpapi_response.get('error', 'Unknown error')}"
            )
        
        organic_results = serpapi_response.get("organic_results", [])
        if not isinstance(organic_results, list) or not organic_results:
            raise HTTPException(
                status_code=404,
                detail="No search results found."
            )
        
        valid_results = []
        for res in organic_results[:SEARCH_RESULTS_LIMIT]:
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
    except httpx.RequestError as e:
        logger.error(f"SerpAPI request error: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Search service unavailable: {str(e)}"
        )
    except httpx.TimeoutException:
        logger.error("SerpAPI request timed out")
        raise HTTPException(
            status_code=504,
            detail="Search request timed out"
        )
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

async def generate_ai_response(system_prompt: str, user_prompt: str, model: str, http_client: httpx.AsyncClient) -> str:
    """
    Generates AI response with improved error handling and performance monitoring.
    """
    try:
        start_time = time.time()
        # Create model request
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        execution_time = time.time() - start_time
        logger.info(f"AI response generation with {model} took {execution_time:.2f}s")
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"AI response generation error with {model}: {str(e)}")
        if "rate limit" in str(e).lower():
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        elif "timeout" in str(e).lower():
            raise HTTPException(
                status_code=504,
                detail="AI response generation timed out"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"AI response generation failed: {str(e)}"
            )

# API Endpoints
@app.post("/legal-assistant/", response_model=ApiResponse)
async def legal_assistant(
    query_input: QueryInput, 
    background_tasks: BackgroundTasks,
    http_client: httpx.AsyncClient = Depends(get_http_client)
):
    start_time = time.time()
    try:
        user_query = query_input.query.strip()

        if not user_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Rate limiting check 
        
        if query_input.web_search:
            #  web search and generate response
            search_results = await perform_search(user_query, http_client)
            
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
                "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                http_client
            )
            
            return {
                "query": user_query, 
                "response": ai_response, 
                "source": "web_search",
                "is_doc_gen": None
            }
        
        else:
            # Generate AI response directly
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
            
            model = "deepseek-ai/DeepSeek-R1" if query_input.deep_think else "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
            
            # Check if model is available
            if not model_status.get(model, False):
                logger.warning(f"Model {model} is not ready, falling back to Qwen")
                model = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
                if not model_status.get(model, False):
                    raise HTTPException(status_code=503, detail="AI models are not ready yet. Please try again later.")
            
            # Run doc_gen check in background to avoid blocking
            background_tasks.add_task(doc_gen, user_query)
            
            # Generate AI response
            ai_response = await generate_ai_response(casebud_system_prompt, user_query, model, http_client)
            
            # Check if document generation is needed (cached result)
            doc = doc_gen(user_query)
            
            logger.info(f"Total request processing time: {time.time() - start_time:.2f}s")
            
            return {
                "query": user_query, 
                "response": ai_response, 
                "source": model.split("/")[-1], 
                "is_doc_gen": doc
            }
            
    except HTTPException:
        # Re-raise HTTP exceptions without logging
        raise
    except Exception as e:
        logger.error(f"Unexpected error in legal_assistant: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/model-status/")
async def get_model_status():
    """
    Returns the current status of AI models.
    """
    return model_status

@app.head("/")
@app.get("/")
def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "running", "message": "Legal AI Assistant is online!"}

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions with detailed responses.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail, "status_code": exc.status_code},
    )

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected exceptions.
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "An internal error occurred. Please try again later.", "status_code": 500},
    )
