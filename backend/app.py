# backend/app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from .core.schemas import AskRequest, AskResponse
from .core.rag.agent import app_graph

# --- Application Setup ---
app = FastAPI(
    title="Maths TA API",
    description="An API for the AI Maths Tutor",
    version="0.1.1"
)

# --- CORS Middleware ---
origins = [
    "http://localhost:8501",  # Default for local Streamlit
    "http://localhost:8000", 
    "http://127.0.0.1:3000", # <-- ADD THIS for your HTML/JS frontend
    "http://localhost:3000",  
    # Add your deployed frontend URL here
    # e.g., "https://maths-ta-frontend.onrender.com" 
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Maths TA API"}


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Receives a question with a session_id, invokes the LangGraph agent,
    and returns the final answer with sources.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id cannot be empty.")

    config = {"configurable": {"thread_id": request.session_id}}
    inputs = {"query": request.query, "image_data": request.image_data, "chat_history": []}

    try:
        # Use ainvoke for async compatibility with FastAPI
        final_state = await app_graph.ainvoke(inputs, config=config)
        
        # With a checkpointer, the final state of invoke is the output of the *last* node.
        # We need to get the full state to have all the data.
        full_conversation_state = await app_graph.aget_state(config)
        
        return AskResponse(
            answer=full_conversation_state.values.get("response", "No response generated."),
            sources=full_conversation_state.values.get("sources", [])
        )
    except Exception as e:
        print(f"API Error: An error occurred in the /ask endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")