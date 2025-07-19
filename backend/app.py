from fastapi import FastAPI, HTTPException
from .core.schemas import AskRequest, AskResponse
from .core.rag.agent import get_rag_chain

app = FastAPI(
    title="Maths TA API",
    description="An API for the AI Maths Tutor",
    version="0.1.0"
)

rag_chain = get_rag_chain()
@app.get("/")
def read_root():
    return {"message": "Welcome to the Maths TA API"}

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """Receives a question, invokes the RAG chain, and returns the answer with sources."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        # The chain's .invoke() method takes the input and returns a dictionary
        result = rag_chain.invoke(request.query)
        # We can directly pass this dictionary to the AskResponse model
        return AskResponse(**result)
    except Exception as e:
        print(f"API Error: An error occurred in the /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")
