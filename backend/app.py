from fastapi import FastAPI, HTTPException
from .core.schemas import AskRequest, AskResponse
from .core.rag.agent import app_graph, _get_sources

app = FastAPI(
    title="Maths TA API",
    description="An API for the AI Maths Tutor",
    version="0.1.1"
)

# No agent initialization needed here anymore

@app.get("/")
def read_root():
    return {"message": "Welcome to the Maths TA API"}

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """Receives a question, invokes the RAG chain, and returns the answer with sources."""
    thread_id = "my-test-conversation" 
    
    config = {"configurable": {"thread_id": thread_id}}
    
    inputs = {"query": request.query, "image_data": request.image_data, "chat_history": []}

    try:
        final_state = app_graph.invoke(inputs, config=config)
        
        # Format the final response from the graph's state
        full_conversation_state = app_graph.get_state(config)
        
        return AskResponse(
            answer=full_conversation_state.values.get("response", "No response."),
            sources=full_conversation_state.values.get("sources", [])
        )
    except Exception as e:
        print(f"API Error: An error occurred in the /ask endpoint: {e}")
        # Add traceback for easier debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal error occurred.")