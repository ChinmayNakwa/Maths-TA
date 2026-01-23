# backend/app.py

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage, AIMessage
from contextlib import asynccontextmanager

from .core.schemas import AskRequest, AskResponse
from .core.rag.agent import app_graph
from .utils.voice_service import transcribe_audio, text_to_speech
from .database.setup_db import create_database_if_missing
from .database.postgres import pool, checkpointer, delete_thread_data

# --- Lifecycle Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Startup Sequence ---")
    
    # 1. Ensure the Database exists (using db_setup.py)
    await create_database_if_missing()
    
    # 2. Open the connection pool (using postgres.py)
    print("DB: Opening connection pool...")
    await pool.open()
    
    # 3. Create/Verify Tables (using LangGraph checkpointer)
    print("DB: Verifying checkpoint tables...")
    await checkpointer.setup()
    
    print("--- System Ready ---")
    yield
    
    # Shutdown
    print("--- Shutdown Sequence ---")
    await pool.close()

# --- Application Setup ---
app = FastAPI(
    title="Maths TA API",
    description="An API for the AI Maths Tutor",
    version="0.1.2",
    lifespan=lifespan # Attach the lifecycle manager
)

# --- CORS Middleware ---
# (Keep your existing origins list)
origins = [
    "http://localhost:8501",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow GET, POST, OPTIONS etc.
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the Maths TA API with Persistent History"}

@app.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    """
    Retrieves the chat history for a specific session from Postgres.
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # Get the current state of the graph for this thread
        state_snapshot = await app_graph.aget_state(config)
        
        # If no state exists (empty session), return empty list
        if not state_snapshot.values:
            return {"history": []}

        messages = state_snapshot.values.get("messages", [])
        
        # Format messages for the frontend
        formatted_history = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            
            # Handle content (it might be a list if it contains images)
            content = msg.content
            if isinstance(content, list):
                # Extract text part from complex content (e.g., text + image_url)
                text_parts = [p["text"] for p in content if p.get("type") == "text"]
                content = " ".join(text_parts)
            
            formatted_history.append({
                "role": role,
                "content": content,
                # Optional: Add timestamp if available in metadata
            })
            
        return {"history": formatted_history}

    except Exception as e:
        print(f"Error retrieving history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    # ... (Keep your existing implementation exactly the same) ...
    # The app_graph is already compiled with the Postgres checkpointer,
    # so calling ainvoke automatically saves to Postgres.
    
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id cannot be empty.")

    config = {"configurable": {"thread_id": request.session_id}}
    
    # Handle Image/Text content structure
    message_content = [{"type": "text", "text": request.query}]
    if request.image_data:
        message_content.append(
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{request.image_data}"
            }
        )
    
    inputs = {
        "messages": [HumanMessage(content=message_content)],
        "query": request.query,
        "image_data": request.image_data
    }

    try:
        final_state = await app_graph.ainvoke(inputs, config=config)
        
        return AskResponse(
            answer=final_state.get("response", "No response generated."),
            sources=final_state.get("sources", [])
        )
    except Exception as e:
        print(f"API Error: An error occurred in the /ask endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# ... (Keep /transcribe and /speak endpoints) ...

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
    message_content = [{"type": "text", "text": request.query}]
    if request.image_data:
        message_content.append(
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{request.image_data}"
            }
        )
    
    inputs = {
        "messages": [HumanMessage(content=message_content)],
        "query": request.query,
        "image_data": request.image_data
    }


    try:
        # Use ainvoke for async compatibility with FastAPI
        final_state = await app_graph.ainvoke(inputs, config=config)
    
        
        return AskResponse(
            answer=final_state.get("response", "No response generated."),
            sources=final_state.get("sources", [])
        )
    except Exception as e:
        print(f"API Error: An error occurred in the /ask endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
    
@app.post("/transcribe")
async def handle_transcribe(audio_file: UploadFile = File(...)):
    """
    Receives an audio file, transcribes it, and returns the text.
    """
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided.")
    
    audio_bytes = await audio_file.read()
    transcribed_text = transcribe_audio(audio_bytes)
    return {"transcription": transcribed_text}

@app.post("/speak")
async def handle_speak(text: str = Form(...)):
    """
    Receives text, converts it to speech, and streams back the audio.
    """
    audio_generator = text_to_speech(text)
    # The local model generates WAV audio
    return StreamingResponse(audio_generator, media_type="audio/wav")

@app.delete("/history/{session_id}")
async def delete_chat_history(session_id: str):
    """
    Deletes a specific chat session from the database.
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    
    try:
        await delete_thread_data(session_id)
        return {"message": "Chat history deleted successfully"}
    except Exception as e:
        print(f"Error deleting history: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete chat history")