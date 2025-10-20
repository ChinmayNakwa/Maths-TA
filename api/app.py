# backend/app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from langchain_core.messages import HumanMessage
from .core.schemas import AskRequest, AskResponse
from .core.rag.agent import app_graph
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from .utils.voice_service import transcribe_audio, text_to_speech

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