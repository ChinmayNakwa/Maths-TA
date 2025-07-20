# backend/core/rag/agent.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from typing import List, Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
# FIX: Replace this line:
# from langchain_core.pydantic_v1 import BaseModel, Field
# With this:
from pydantic import BaseModel, Field
import operator
import json
import os
import re

# Our own modules
from backend.database.astra_db_connection import get_vector_store
from backend.core.schemas import SourceDocument
from backend.config import settings
from langchain.docstore.document import Document
from backend.core.rag.rag_manager import RAGManager

# LangSmith Configuration
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", settings.LANGSMITH_API_KEY)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", settings.LANGSMITH_PROJECT)

# --- HELPER FUNCTIONS ---

def _format_docs(docs: List[Document]) -> str:
    """
    Prepares the retrieved documents for insertion into the prompt.
    Enhanced version with visual context handling.
    """
    if not docs:
        return "No context was retrieved from the knowledge base."

    formatted_docs = []
    image_urls = set()

    for i, doc in enumerate(docs):
        source_name = doc.metadata.get('source', 'unknown')
        location_text = f"Source {i+1}"

        if source_name in ["book", "PROBABILITY_FOR_COMPUTER_SCIENTIST"]:
            page_num = doc.metadata.get('page_number', 'N/A')
            location_text = f"Context from Book (Page {page_num})"
        elif source_name == "video":
            title = doc.metadata.get('title', 'N/A')
            try:
                ss = int(doc.metadata.get('start_time_sec', 0))
                h, m = divmod(ss, 3600)
                m, s = divmod(m, 60)
                timestamp = f"{h:02d}:{m:02d}:{s:02d}"
                location_text = f"Context from Video '{title}' (at {timestamp})"
            except (ValueError, TypeError):
                location_text = f"Context from Video '{title}'"

        doc_string = f"--- {location_text} ---\n{doc.page_content}"
        formatted_docs.append(doc_string)

        # Collect image URLs
        if doc.metadata.get("image_path"):
            image_urls.add(doc.metadata.get("image_path"))

    final_context_str = "\n\n".join(formatted_docs)
    
    # Add visual context if available
    if image_urls:
        final_context_str += "\n\n--- Associated Visual Context (URLs) ---\n"
        final_context_str += "\n".join(f"- {url}" for url in image_urls)
            
    return final_context_str


def _get_sources(docs: List[Document]) -> List[SourceDocument]:
    """
    Creates a list of SourceDocument objects from the retrieved docs.
    Enhanced version with better source handling.
    """
    sources = []
    unique_sources = set()

    for doc in docs:
        source_name = doc.metadata.get('source', 'unknown')
        source_type = 'unknown'
        location = 'Unknown'
        url = None
        source_key = None
        
        if source_name in ["book", "PROBABILITY_FOR_COMPUTER_SCIENTIST"]:
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
                h, m = divmod(ss, 3600)
                m, s = divmod(m, 60)
                timestamp = f"{h:02d}:{m:02d}:{s:02d}"
                location = f"'{title}' at {timestamp}"
            except (ValueError, TypeError):
                location = f"'{title}'"
            source_key = f"video_{doc.metadata.get('video_id')}"

        if source_key and source_key not in unique_sources:
            sources.append(SourceDocument(source_type=source_type, location=location, url=url))
            unique_sources.add(source_key)
            
    return sources


def _filter_docs_by_title(docs: List[Document], title_keyword: str) -> List[Document]:
    """
    Client-side filtering for title keywords since Astra DB doesn't support regex.
    """
    if not title_keyword:
        return docs
    
    filtered_docs = []
    title_lower = title_keyword.lower()
    
    for doc in docs:
        doc_title = doc.metadata.get('title', '').lower()
        if title_lower in doc_title:
            filtered_docs.append(doc)
    
    return filtered_docs


# --- PYDANTIC MODELS ---

class Critique(BaseModel):
    """The model's internal critique of the student's work."""
    critique: str = Field(description="A brief, internal-monologue style analysis of the student's work.")
    needs_context: str = Field(description="A search query for the textbook if more information is needed, otherwise an empty string ''.")


class RouteQuery(BaseModel):
    """Routes the user's query to the appropriate path."""
    route: str = Field(description="Set to 'critique' if the user is asking for help with their work, a specific problem, or has uploaded an image. Otherwise, set to 'rag'.")


class SearchFilters(BaseModel):
    """Structured filters extracted from the user's query for a targeted search."""
    title_keyword: Optional[str] = Field(None, description="A specific keyword to search for in the video title, like 'Lecture 4' or 'Lecture 22'.")
    page_number: Optional[int] = Field(None, description="A specific page number mentioned in the query.")


# --- GRAPH STATE ---

class GraphState(TypedDict):
    """
    Enhanced state with proper routing and filtering support.
    """
    query: str
    image_data: Optional[str]
    route: str
    filters: Dict
    transcribed_work: str
    context: List[Document]
    critique: Dict
    response: str
    sources: List[SourceDocument]
    chat_history: List[tuple]


# --- TUTOR AGENT CLASS ---

class TutorAgent:
    def __init__(self):
        self.rag_manager = RAGManager()
        # Use updated models with appropriate temperatures
        self.vision_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0, 
            api_key=settings.GOOGLE_API_KEY
        )
        self.reasoning_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=1, 
            api_key=settings.GOOGLE_API_KEY
        )
        # Create structured output chains
        self.filter_extraction_model = self.reasoning_model.with_structured_output(SearchFilters)
        self.router_chain = self.reasoning_model.with_structured_output(RouteQuery)
        self.critique_chain = self.reasoning_model.with_structured_output(Critique)
        print("TutorAgent: Initialized successfully.")

    def extract_filters(self, state: GraphState) -> dict:
        """Node: Extracts structured filters from the user's query."""
        print("---NODE: EXTRACT FILTERS---")
        query = state["query"]
        
        system_prompt = """You are an expert at parsing user questions. Your task is to extract specific identifiers that can be used to filter a database search.
        If the user mentions 'lecture 4', extract `title_keyword: "Lecture 4"`.
        If the user mentions 'lecture twenty-two', extract `title_keyword: "Lecture 22"`.
        If the user asks about 'page 61', extract `page_number: 61`.
        If no specific identifiers are mentioned, do not extract any filters.
        """
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt), 
                ("human", "{question}")
            ])
            chain = prompt | self.filter_extraction_model
            filter_object = chain.invoke({"question": query})
            extracted_filters = {k: v for k, v in filter_object.dict().items() if v is not None}
            print(f"Extracted Filters: {extracted_filters}")
            return {"filters": extracted_filters}
        except Exception as e:
            print(f"Error extracting filters: {e}")
            return {"filters": {}}

    def route_question(self, state: GraphState) -> dict:
        """
        Node: Determines whether to use simple RAG or the full critique workflow.
        Enhanced with better routing logic.
        """
        print("---NODE: ROUTE QUESTION---")
        query = state["query"]
        image_data = state.get("image_data")

        # If image is provided, always route to critique
        if image_data:
            print("Image provided. Routing to critique.")
            return {"route": "critique"}

        # Use LLM to determine route based on query content
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at routing user questions. Your goal is to determine if the user is asking:
                
                1. A general question about concepts (route: 'rag') - Examples:
                   - "What is Bayes theorem?"
                   - "Explain probability distributions"
                   - "How do I calculate variance?"
                
                2. Asking for help with specific work/problem (route: 'critique') - Examples:
                   - "Can you check my work?"
                   - "I'm stuck on this problem"
                   - "Is my solution correct?"
                   - "Help me with this calculation"
                
                Choose 'critique' if they want help with their work, otherwise choose 'rag'."""),
                ("human", "User question: {question}")
            ])
            
            chain = prompt | self.router_chain
            result = chain.invoke({"question": query})
            route = result.route
            
        except Exception as e:
            print(f"Error in routing, defaulting to RAG: {e}")
            # Default to RAG if routing fails
            route = "rag"

        print(f"Routing decision: {route}")
        return {"route": route}

    def simple_rag(self, state: GraphState) -> dict:
        """
        Node: Handles simple Q&A with filtering support.
        Fixed to work with Astra DB limitations.
        """
        print("---NODE: SIMPLE RAG (WITH FILTERS)---")
        query = state["query"]
        filters = state.get("filters", {})
        
        if not query:
            print("No query provided.")
            return {"response": "I don't see a question to answer.", "sources": []}
        
        try:
            # Build Astra DB filter - only use supported operators
            astra_filter = {}
            title_keyword = None
            
            # Handle page number filter (supported by Astra DB)
            if "page_number" in filters: 
                astra_filter["page_number"] = filters["page_number"]
                print(f"Applying page filter: {astra_filter}")
            
            # Handle title keyword - we'll do client-side filtering
            if "title_keyword" in filters:
                title_keyword = filters["title_keyword"]
                print(f"Will apply client-side title filter: {title_keyword}")
            
            # Create retriever with supported filters only
            search_kwargs = {"k": 10}  # Get more docs for client-side filtering
            if astra_filter:
                search_kwargs["filter"] = astra_filter
            
            retriever = self.rag_manager.vector_store.as_retriever(search_kwargs=search_kwargs)
            docs = retriever.invoke(query)
            
            # Apply client-side title filtering if needed
            if title_keyword:
                docs = _filter_docs_by_title(docs, title_keyword)
                print(f"After title filtering: {len(docs)} documents")
            
            # Limit to top 5 results after filtering
            docs = docs[:5]
            
            sources = _get_sources(docs)
            context_str = _format_docs(docs)

            # Create RAG prompt
            template = """You are a teaching assistant at Stanford in probability and statistics. You must answer the user's question based *only* on the provided context.
            The context contains snippets from a textbook and video lectures. Synthesize the information from all context snippets to provide a comprehensive and clear answer.
            If the context is insufficient, state that you cannot answer from the given information.

            CONTEXT:
            {context}

            USER QUESTION:
            {question}

            ANSWER:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.reasoning_model | StrOutputParser()
            answer = chain.invoke({"context": context_str, "question": query})
            
            return {"response": answer, "sources": sources}
            
        except Exception as e:
            print(f"Error in simple RAG: {e}")
            return {
                "response": "I encountered an error while retrieving information. Please try rephrasing your question.",
                "sources": []
            }

    def transcribe_image(self, state: GraphState) -> dict:
        """
        Node: Transcribes the user's work from an image if it exists.
        Enhanced with better error handling.
        """
        print("---NODE: TRANSCRIBE IMAGE---")
        image_data = state.get("image_data")
        
        if not image_data:
            print("No image provided, but continuing critique path.")
            return {"transcribed_work": "No image was provided."}
        
        try:
            print("Transcribing image...")
            prompt = [
                {
                    "type": "text", 
                    "text": "This image contains a student's handwritten work for a math problem. Transcribe the student's steps exactly as written, including any equations, calculations, or diagrams. Focus only on the mathematical work, not the surrounding environment."
                },
                {
                    "type": "image_url", 
                    "image_url": f"data:image/jpeg;base64,{image_data}"
                }
            ]
            
            response = self.vision_model.invoke(prompt)
            transcribed_work = response.content
            print(f"Transcribed Work: {transcribed_work}")
            return {"transcribed_work": transcribed_work}
            
        except Exception as e:
            print(f"Error transcribing image: {e}")
            return {"transcribed_work": "Failed to transcribe the image. Please describe your work in text."}

    def critique_work(self, state: GraphState) -> dict:
        """
        Node: Analyzes the user's work and decides if external context is needed.
        FIXED: Enhanced with robust error handling and corrected prompt template.
        """
        print("---NODE: CRITIQUE WORK---")
        query = state["query"]
        transcribed_work = state["transcribed_work"]
        
        try:
            # FIXED: Corrected the prompt template - no variable placeholders in JSON format
            system_prompt = """You are a Math Teaching Assistant at Stanford. Your task is to analyze a student's question and their work.

            First, determine if the student's approach is on the right track.
            Second, identify the specific concept they might be struggling with.
            Third, decide if you need to look up information from the textbook or video lectures to help them.
            
            You must respond in this exact JSON format:
            {{
                "critique": "A brief analysis of the student's work and approach",
                "needs_context": "A search query for the textbook/video lectures if more information is needed, otherwise an empty string"
            }}
            
            Example response:
            {{
                "critique": "Student is attempting to solve a probability problem but made an error in calculating combinations. The approach is correct but needs clarification on the combination formula.",
                "needs_context": "probability combinations formula"
            }}"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Student Question: {query}\n\nStudent's Transcribed Work:\n{transcribed_work}")
            ])
            
            # Try multiple approaches for structured output
            try:
                # Method 1: with_structured_output (preferred)
                chain = prompt | self.critique_chain
                critique_object = chain.invoke({"query": query, "transcribed_work": transcribed_work})
                critique_json = critique_object.dict()
                
            except Exception as e:
                print(f"Structured output failed, trying JsonOutputParser: {e}")
                
                # Method 2: JsonOutputParser fallback
                try:
                    parser = JsonOutputParser(pydantic_object=Critique)
                    chain = prompt | self.reasoning_model | parser
                    critique_json = chain.invoke({"query": query, "transcribed_work": transcribed_work})
                    
                except Exception as e2:
                    print(f"JsonOutputParser failed, using manual parsing: {e2}")
                    
                    # Method 3: Manual parsing with string extraction
                    chain = prompt | self.reasoning_model | StrOutputParser()
                    response_text = chain.invoke({"query": query, "transcribed_work": transcribed_work})
                    critique_json = self._parse_json_response(response_text, query)
            
        except Exception as e:
            print(f"All critique methods failed: {e}")
            # Final fallback
            critique_json = {
                "critique": f"Unable to analyze the work due to a processing error. The question appears to be about: {query[:100]}...",
                "needs_context": "probability mathematics concepts"
            }
        
        print(f"Critique: {critique_json}")
        return {"critique": critique_json}

    def _parse_json_response(self, response_text: str, fallback_query: str) -> dict:
        """Helper method to parse JSON from LLM response with multiple fallback strategies."""
        try:
            # Try direct JSON parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            try:
                # Try to extract JSON from markdown code blocks
                json_pattern = r'```json\s*(.*?)\s*```'
                matches = re.findall(json_pattern, response_text, re.DOTALL)
                if matches:
                    return json.loads(matches[0])
                else:
                    # Try to find JSON-like content
                    json_pattern = r'\{.*?\}'
                    matches = re.findall(json_pattern, response_text, re.DOTALL)
                    if matches:
                        return json.loads(matches[0])
                    else:
                        raise json.JSONDecodeError("No JSON found", response_text, 0)
            except json.JSONDecodeError:
                # Create structured response from text
                print(f"Could not parse JSON from response: {response_text[:200]}...")
                return {
                    "critique": response_text[:300] + "..." if len(response_text) > 300 else response_text,
                    "needs_context": " ".join(fallback_query.split()[:5])  # Use first few words as search query
                }

    def retrieve_context(self, state: GraphState) -> dict:
        """
        Node: Retrieves context from the vector store with filtering if needed for the critique.
        Fixed to work with Astra DB limitations.
        """
        print("---NODE: RETRIEVE CONTEXT (WITH FILTERS)---")
        critique_search_query = state.get("critique", {}).get("needs_context", "")
        
        if not critique_search_query:
            print("No context needed. Skipping retrieval.")
            return {"context": [], "sources": []}
        
        try:
            filters = state.get("filters", {})
            astra_filter = {}
            title_keyword = None
            
            # Only use supported Astra DB filters
            if "page_number" in filters: 
                astra_filter["page_number"] = filters["page_number"]
            
            # Handle title keyword with client-side filtering
            if "title_keyword" in filters:
                title_keyword = filters["title_keyword"]

            print(f"Retrieving context for query: '{critique_search_query}' with DB filter: {astra_filter}")
            
            search_kwargs = {"k": 6}  # Get more for client-side filtering
            if astra_filter:
                search_kwargs["filter"] = astra_filter
            
            retriever = self.rag_manager.vector_store.as_retriever(search_kwargs=search_kwargs)
            docs = retriever.invoke(critique_search_query)
            
            # Apply client-side title filtering if needed
            if title_keyword:
                docs = _filter_docs_by_title(docs, title_keyword)
                print(f"After title filtering: {len(docs)} documents")
            
            # Limit to top 3 results
            docs = docs[:3]
            sources = _get_sources(docs)
            return {"context": docs, "sources": sources}
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return {"context": [], "sources": []}

    def generate_socratic_response(self, state: GraphState) -> dict:
        """
        Node: Generates the final Socratic-style hint for the user.
        Enhanced with better prompt and chat history handling.
        """
        print("---NODE: GENERATE SOCRATIC RESPONSE---")
        query = state["query"]
        transcribed_work = state["transcribed_work"]
        critique = state.get("critique", {}).get("critique", "")
        context = state.get("context", [])
        chat_history = state.get("chat_history", [])
        context_str = _format_docs(context)
        
        system_prompt = """You are a helpful and encouraging AI Math Teaching Assistant at Stanford. Your role is to guide students through mathematical problems step-by-step using the Socratic method.

        Guidelines:
        - Be encouraging and supportive - acknowledge what the student did well
        - Point out specific errors or misconceptions gently
        - Ask guiding questions rather than giving direct answers
        - Provide hints that lead students to discover solutions themselves
        - Use context from textbooks/videos to support your guidance
        - Reference relevant formulas or concepts from the provided context
        - Build on the conversation history to avoid repetition
        
        Your goal is to help students learn and understand, not just get the right answer."""
        
        # Build conversation with history
        prompt_messages = [("system", system_prompt)]
        
        # Add chat history
        for human, ai in chat_history:
            prompt_messages.append(("human", human))
            prompt_messages.append(("ai", ai))
        
        # Add current interaction
        prompt_messages.append((
            "human", 
            "My Question: {question}\n\nMy Work:\n{work}\n\nYour Internal Analysis:\n{critique}\n\nRelevant Context from Textbook/Videos:\n{context}"
        ))
        
        try:
            prompt_template = ChatPromptTemplate.from_messages(prompt_messages)
            chain = prompt_template | self.reasoning_model | StrOutputParser()
            response = chain.invoke({
                "question": query, 
                "work": transcribed_work, 
                "critique": critique, 
                "context": context_str
            })
            print(f"Generated Response: {response}")
            return {"response": response}
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return {"response": "I encountered an error while preparing my response. Could you please try asking your question again?"}


# --- EDGE FUNCTIONS ---

def should_go_to_critique_or_rag(state: GraphState) -> str:
    """Enhanced routing decision based on state."""
    print("---EDGE: CRITIQUE OR RAG?---")
    route = state.get("route", "")
    
    if route == "critique":
        print("Decision: Route to critique workflow.")
        return "transcribe_image"
    else:
        print("Decision: Route to simple RAG.")
        return "simple_rag"


def should_retrieve_context(state: GraphState) -> str:
    """Decides whether to retrieve additional context."""
    print("---EDGE: SHOULD RETRIEVE?---")
    needs_context = state.get("critique", {}).get("needs_context", "")
    
    if needs_context:
        print("Decision: Yes, context is needed.")
        return "retrieve_context"
    else:
        print("Decision: No, context is not needed.")
        return "generate_socratic_response"


# --- GRAPH ASSEMBLY ---

# Instantiate the agent
agent_logic = TutorAgent()

# Create the workflow graph
workflow = StateGraph(GraphState)

# Add all nodes
workflow.add_node("extract_filters", agent_logic.extract_filters)
workflow.add_node("route_question", agent_logic.route_question)
workflow.add_node("simple_rag", agent_logic.simple_rag)
workflow.add_node("transcribe_image", agent_logic.transcribe_image)
workflow.add_node("critique_work", agent_logic.critique_work)
workflow.add_node("retrieve_context", agent_logic.retrieve_context)
workflow.add_node("generate_socratic_response", agent_logic.generate_socratic_response)

# Define the workflow structure
workflow.set_entry_point("extract_filters")
workflow.add_edge("extract_filters", "route_question")

# Route from question analysis
workflow.add_conditional_edges(
    "route_question",
    should_go_to_critique_or_rag,
    {
        "transcribe_image": "transcribe_image",
        "simple_rag": "simple_rag"
    }
)

# Simple RAG path ends immediately
workflow.add_edge("simple_rag", END)

# Critique workflow path
workflow.add_edge("transcribe_image", "critique_work")
workflow.add_conditional_edges(
    "critique_work",
    should_retrieve_context,
    {
        "retrieve_context": "retrieve_context",
        "generate_socratic_response": "generate_socratic_response"
    }
)
workflow.add_edge("retrieve_context", "generate_socratic_response")
workflow.add_edge("generate_socratic_response", END)

# Add memory for conversation persistence
memory = MemorySaver()

# Compile the graph
app_graph = workflow.compile(checkpointer=memory)

print("Combined LangGraph agent compiled successfully with Astra DB compatible filtering, intelligent routing, memory, and robust error handling.")