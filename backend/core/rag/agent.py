# backend/core/rag/agent.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.pydantic_v1 import BaseModel, Field
import operator
import json
import os

# Our own modules
from backend.database.astra_db_connection import get_vector_store
from backend.core.schemas import SourceDocument
from backend.config import settings
from langchain.docstore.document import Document
from backend.core.rag.rag_manager import RAGManager

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Maths-TA"

# --- HELPER FUNCTIONS (must be outside the class) ---

def _format_docs(docs: List[Document]) -> str:
    """
    Prepares the retrieved documents for insertion into the prompt.
    This version defensively handles different source names.
    """
    if not docs:
        return "No context was retrieved from the knowledge base."

    formatted_docs = []
    image_urls = set()

    for i, doc in enumerate(docs):
        source_name = doc.metadata.get('source', 'unknown')
        location_text = f"Source {i+1}"

        if source_name == "book" or source_name == "PROBABILITY_FOR_COMPUTER_SCIENTIST":
            page_num = doc.metadata.get('page_number', 'N/A')
            location_text = f"Context from Book (Page {page_num})"
        elif source_name == "video":
            title = doc.metadata.get('title', 'N/A')
            try:
                ss = int(doc.metadata.get('start_time_sec', 0))
                h, m = divmod(ss, 3600); m, s = divmod(m, 60)
                timestamp = f"{h:02d}:{m:02d}:{s:02d}"
                location_text = f"Context from Video '{title}' (at {timestamp})"
            except (ValueError, TypeError):
                location_text = f"Context from Video '{title}'"

        doc_string = f"--- {location_text} ---\n{doc.page_content}"
        formatted_docs.append(doc_string)

        if doc.metadata.get("image_path"):
            image_urls.add(doc.metadata.get("image_path"))

    final_context_str = "\n\n".join(formatted_docs)
    
    if image_urls:
        final_context_str += "\n\n--- Associated Visual Context (URLs) ---\n"
        final_context_str += "\n".join(f"- {url}" for url in image_urls)
            
    return final_context_str


def _get_sources(docs: List[Document]) -> List[SourceDocument]:
    """
    Creates a list of SourceDocument objects from the retrieved docs.
    This version defensively handles different source names.
    """
    sources = []
    unique_sources = set()

    for doc in docs:
        source_name = doc.metadata.get('source', 'unknown')
        source_type = 'unknown'; location = 'Unknown'; url = None; source_key = None
        
        if source_name == "book" or source_name == "PROBABILITY_FOR_COMPUTER_SCIENTIST":
            source_type = 'book'
            page_num = doc.metadata.get('page_number', 'N/A')
            location = f"Page {page_num}"
            url = doc.metadata.get("image_path")
            source_key = f"book_page_{page_num}"
            
        elif source_name == 'video':
            source_type = 'video'
            title = doc.metadata.get('title', 'N/A')
            url = doc.metadata.get("video_url")
            try:
                ss = int(doc.metadata.get('start_time_sec', 0))
                h, m = divmod(ss, 3600); m, s = divmod(m, 60)
                timestamp = f"{h:02d}:{m:02d}:{s:02d}"
                location = f"'{title}' at {timestamp}"
            except (ValueError, TypeError):
                location = f"'{title}'"
            source_key = f"video_{doc.metadata.get('video_id')}"

        if source_key and source_key not in unique_sources:
            sources.append(SourceDocument(source_type=source_type, location=location, url=url))
            unique_sources.add(source_key)
            
    return sources


# --- 1. Define the State for our Graph (with memory) ---

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        query: The user's current question.
        image_data: The base64-encoded image from the current turn.
        transcribed_work: Text extracted from the image in the current turn.
        context: Documents retrieved in the current turn.
        critique: LLM's analysis of the user's work in the current turn.
        response: The final response for the current turn.
        sources: A list of sources used in the current turn.
        chat_history: A list of (human_message, ai_message) tuples from previous turns.
    """
    query: str
    image_data: str
    transcribed_work: str
    context: List[Document]
    critique: Dict
    response: str
    sources: List[SourceDocument]
    chat_history: List[tuple]

class Critique(BaseModel):
    """
    The model's internal critique of the student's work.
    """
    critique: str = Field(description="A brief, internal-monologue style analysis of the student's work.")
    needs_context: str = Field(description="A search query for the textbook if more information is needed, otherwise an empty string ''.")

class TutorAgent:
    def __init__(self):
        self.rag_manager = RAGManager()
        self.retriever = self.rag_manager.vector_store.as_retriever(search_kwargs={"k":3})
        self.vision_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=settings.GOOGLE_API_KEY)
        self.reasoning_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, api_key=settings.GOOGLE_API_KEY)
        print("TutorAgent: Initialized successfully.")

    def transcribe_image(self, state: GraphState) -> dict:
        print("---NODE: TRANSCRIBE IMAGE---")
        image_data = state.get("image_data")
        if not image_data:
            print("No image provided. Skipping transcription.")
            return {"transcribed_work": ""}
        print("Transcribing image...")
        prompt = [
            {"type": "text", "text": "This image contains a student's handwritten work for a math problem. Transcribe the student's steps exactly as written. Focus only on the work, not the surrounding environment."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_data}"}
        ]
        response = self.vision_model.invoke(prompt)
        transcribed_work = response.content
        print(f"Transcribed Work: {transcribed_work}")
        return {"transcribed_work": transcribed_work}

    def critique_work(self, state: GraphState) -> dict:
        """
        Node: Analyzes the user's work and decides if external context is needed.
        Fixed version with proper structured output handling.
        """
        print("---NODE: CRITIQUE WORK---")
        query = state["query"]
        transcribed_work = state["transcribed_work"]
        
        try:
            system_prompt = """You are a Math Teaching Assistant at Stanford. Your task is to analyze a student's question and their work.
            First, determine if the student's approach is on the right track.
            Second, identify the specific concept they might be struggling with.
            Third, decide if you need to look up information from the textbook or video lectures to help them.
            
            You must respond with a JSON object containing exactly these fields:
            - "critique": A brief analysis of the student's work
            - "needs_context": A search query for the textbook/video lectures if more information is needed, otherwise an empty string ""
            
            Example response:
            {{
                "critique": "Student is attempting to solve probability but made an error in calculating combinations",
                "needs_context": "probability combinations formula"
            }}
            
            Respond with ONLY valid JSON, no additional text."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", f"Student Question: {query}\n\nStudent's Transcribed Work:\n{transcribed_work}")
            ])
            
            # Option 1: Use with_structured_output (recommended)
            # This should work with newer versions of langchain
            try:
                structured_llm = self.reasoning_model.with_structured_output(Critique)
                chain = prompt | structured_llm
                critique_object = chain.invoke({})
                critique_json = critique_object.dict()
            
            except AttributeError:
                # Option 2: Fallback to JsonOutputParser if with_structured_output is not available
                print("with_structured_output not available, using JsonOutputParser")
                parser = JsonOutputParser(pydantic_object=Critique)
                prompt = prompt.partial(format_instructions=parser.get_format_instructions())
                chain = prompt | self.reasoning_model | parser
                critique_json = chain.invoke({})
            
        except Exception as e:
            print(f"Error in structured output, falling back to manual parsing: {e}")
            
            # Option 3: Fallback to string parsing with manual JSON extraction
            system_prompt = """You are an expert Math Professor at Stanford. Analyze the student's work and respond with JSON only.
            
            Respond with exactly this JSON format (no additional text):
            {{
                "critique": "your analysis here",
                "needs_context": "search query or empty string"
            }}"""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", f"Student Question: {query}\n\nStudent's Transcribed Work:\n{transcribed_work}")
            ])
            
            chain = prompt | self.reasoning_model | StrOutputParser()
            response_text = chain.invoke({})
            
            # Try to parse JSON from the response
            try:
                # Try direct JSON parsing
                critique_json = json.loads(response_text)
            except json.JSONDecodeError:
                try:
                    # Try to extract JSON from markdown code blocks
                    import re
                    json_pattern = r'```json\s*(.*?)\s*```'
                    matches = re.findall(json_pattern, response_text, re.DOTALL)
                    if matches:
                        critique_json = json.loads(matches[0])
                    else:
                        # Try to find JSON-like content
                        json_pattern = r'\{.*\}'
                        matches = re.findall(json_pattern, response_text, re.DOTALL)
                        if matches:
                            critique_json = json.loads(matches[0])
                        else:
                            raise json.JSONDecodeError("No JSON found", response_text, 0)
                            
                except json.JSONDecodeError:
                    # Final fallback: create structured response
                    print(f"Could not parse JSON from response: {response_text}")
                    critique_json = {
                        "critique": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                        "needs_context": "probability mathematics concepts"  # Default search query
                    }
        
        print(f"Critique: {critique_json}")
        return {"critique": critique_json}
    
    def retrieve_context(self, state: GraphState) -> dict:
        print("---NODE: RETRIEVE CONTEXT---")
        search_query = state.get("critique", {}).get("needs_context", "")
        if not search_query:
            print("No context needed. Skipping retrieval.")
            return {"context": [], "sources": []}
        print(f"Retrieving context for query: {search_query}")
        docs = self.retriever.invoke(search_query)
        sources = _get_sources(docs)
        return {"context": docs, "sources": sources}

    def generate_response(self, state: GraphState) -> dict:
        print("---NODE: GENERATE RESPONSE---")
        query = state["query"]
        transcribed_work = state["transcribed_work"]
        critique = state.get("critique", {}).get("critique", "")
        context = state.get("context", [])
        chat_history = state.get("chat_history", [])
        context_str = _format_docs(context)
        
        system_prompt = """You are a helpful and encouraging AI Maths TA. Your role is to guide students through mathematical problems step-by-step, helping them understand concepts rather than just providing answers.

        Guidelines:
        - Be encouraging and supportive
        - Point out what the student did well
        - Identify specific errors or misconceptions
        - Provide hints and guidance rather than complete solutions
        - Use the context from textbooks/videos when available
        - Reference relevant formulas or concepts from the provided context
        
        You have also been provided with the history of your conversation with the student. Use it to avoid repeating yourself and to understand the context of their latest message."""
        
        prompt_messages = [("system", system_prompt)]
        for human, ai in chat_history:
            prompt_messages.append(("human", human))
            prompt_messages.append(("ai", ai))
        prompt_messages.append(("human", "My Question: {question}\n\nMy Work:\n{work}\n\nYour Internal Analysis:\n{critique}\n\nRelevant Context from Textbook/Videos:\n{context}"))
        
        prompt_template = ChatPromptTemplate.from_messages(prompt_messages)
        chain = prompt_template | self.reasoning_model | StrOutputParser()
        response = chain.invoke({"question": query, "work": transcribed_work, "critique": critique, "context": context_str})
        print(f"Generated Response: {response}")
        return {"response": response}

# --- 3. Define the Graph Edges ---
def should_retrieve_context(state: GraphState) -> str:
    print("---EDGE: SHOULD RETRIEVE?---")
    needs_context = state.get("critique", {}).get("needs_context", "")
    if needs_context:
        print("Decision: Yes, context is needed.")
        return "retrieve_context"
    else:
        print("Decision: No, context is not needed.")
        return "generate_response"

# --- 4. Assemble the Graph ---

# Instantiate the agent class containing the node logic
agent_logic = TutorAgent()

# Define the graph structure
workflow = StateGraph(GraphState)

# Add the nodes
workflow.add_node("transcribe_image", agent_logic.transcribe_image)
workflow.add_node("critique_work", agent_logic.critique_work)
workflow.add_node("retrieve_context", agent_logic.retrieve_context)
workflow.add_node("generate_response", agent_logic.generate_response)

# Define the workflow edges
workflow.set_entry_point("transcribe_image")
workflow.add_edge("transcribe_image", "critique_work")
workflow.add_conditional_edges(
    "critique_work",
    should_retrieve_context,
    {
        "retrieve_context": "retrieve_context",
        "generate_response": "generate_response"
    }
)
workflow.add_edge("retrieve_context", "generate_response")
workflow.add_edge("generate_response", END)

# Add a checkpointer for memory. MemorySaver is simple and in-memory.
memory = MemorySaver()

# Compile the graph into a runnable object, now with memory
app_graph = workflow.compile(checkpointer=memory)

print("LangGraph agent compiled successfully with memory.")