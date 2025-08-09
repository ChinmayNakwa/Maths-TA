# backend/core/rag/agent.py

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from typing import List, Dict, Any, TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import operator
import json
import os
import re
from langchain_core.messages import AIMessage, HumanMessage
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


class RoutingDecision(BaseModel):
    """Routes the user's query to the appropriate path: rag, critique, or clarify."""
    route: str = Field(description="Set to 'critique' if the user is asking for help with their work, a specific problem, or has uploaded an image. Set to 'rag' for general questions. Set to 'clarify' if the user's query is ambiguous.")
    clarifying_question: Optional[str] = Field(None, description="If the route is 'clarify', this field should contain a question to ask the user to resolve the ambiguity.")

class SearchFilters(BaseModel):
    """Structured filters extracted from the user's query for a targeted search."""
    title_keyword: Optional[str] = Field(None, description="A specific keyword to search for in the video title, like 'Lecture 4' or 'Lecture 22'.")
    page_number: Optional[int] = Field(None, description="A specific page number mentioned in the query.")

class Reflection(BaseModel):
    """The model's reflection on its own generated answer."""
    is_socratic: bool = Field(description="True if the answer uses the Socratic method by asking guiding questions rather than giving a direct solution. False otherwise.")
    is_safe: bool = Field(description="True if the answer is safe, respectful, and on-topic. False if it's harmful or inappropriate.")
    is_final_answer: bool = Field(description="True if the answer is good enough to send to the user. This should be True only if is_socratic and is_safe are both True.")
    feedback_for_improvement: str = Field(description="Constructive feedback for regenerating the answer if is_final_answer is False. Explain what was wrong (e.g., 'The answer was too direct. Rephrase it as a question that guides the student.').")

# --- GRAPH STATE ---

class GraphState(TypedDict):
    """
    Enhanced state with proper routing and filtering support.
    """
    query: str
    image_data: Optional[str]
    route: str
    clarifying_question: Optional[str]
    filters: Dict
    transcribed_work: str
    context: List[Document]
    critique: Dict
    response: str
    sources: List[SourceDocument]
    messages: Annotated[list, add_messages]
    feedback_for_improvement: Optional[str]
    is_final_answer: bool

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
        self.router_chain = self.reasoning_model.with_structured_output(RoutingDecision)
        self.critique_chain = self.reasoning_model.with_structured_output(Critique)
        self.reflection_chain = self.reasoning_model.with_structured_output(Reflection)
        print("TutorAgent: Initialized successfully.")

    # def prepare_inputs(self, state: GraphState) -> dict:
    #     """
    #     Node: Adds the initial user query to the messages list to be persisted.
    #     """
    #     print("---NODE: PREPARE INPUTS---")
    #     return {"messages": [HumanMessage(content=state["query"])]}

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
        Node: Determines the route, now with awareness of the conversation history
        to correctly handle follow-up answers.
        """
        print("---NODE: ROUTE QUESTION (CONVERSATION-AWARE)---")
        query = state["query"]
        messages = state["messages"]
        image_data = state.get("image_data")
        # MODIFICATION: Get the full message history from the state
        messages = state.get("messages", [])

        if image_data:
            print("Image provided. Routing to critique.")
            return {"route": "critique", "clarifying_question": None}

        # If this is the very first message after the initial greeting, use the simple router.
        # The history will have 1 (initial HumanMessage) or 2 messages (Human + initial AI).
        if len(messages) <= 2:
             print("  - First user message. Using standard routing.")
             # The existing logic is fine for the first turn.
             try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are an expert at routing the FIRST user question... (Your existing 'clarify' prompt)"""),
                    ("human", "User question: {question}")
                ])
                chain = prompt | self.router_chain
                result = chain.invoke({"question": query})
                return {"route": result.route, "clarifying_question": result.clarifying_question}
             except Exception as e:
                print(f"Error in initial routing, defaulting to RAG: {e}")
                return {"route": "rag", "clarifying_question": None}

        # MODIFICATION: For subsequent messages, use a conversation-aware router prompt.
        print("  - Follow-up message detected. Using conversational routing.")
        try:
            # This new prompt asks the model to consider the history.
            system_prompt = """You are an expert at routing a user's LATEST message in an ongoing conversation with a Teaching Assistant.
            Analyze the conversation history, then decide the route for the NEWEST message.

            - If the user's new message is an answer to the TA's last question (like '5 times' or 'yes'), continue the previous path. Check the TA's last message to see what the likely path was (e.g., if the TA asked a socratic question, the route is 'critique').
            - If the user asks a NEW general question (e.g., "what is a p-value?"), route to 'rag'.
            - If the user provides a NEW problem to solve, route to 'critique'.
            - If the user's new message is genuinely ambiguous even with history, route to 'clarify'.

            Based on the full conversation, choose the route for the LATEST user message."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
            ])

            chain = prompt | self.router_chain

            # Pass the entire message history to the chain
            result = chain.invoke({"chat_history": messages})

            print(f"Conversational routing decision: {result.route}")
            return {
                "route": result.route,
                "clarifying_question": result.clarifying_question
            }

        except Exception as e:
            print(f"Error in conversational routing, defaulting to critique: {e}")
            # In a conversation, defaulting to the 'critique' path is often safer.
            return {"route": "critique", "clarifying_question": None}
        
    def simple_rag(self, state: GraphState) -> dict:
        """
        Node: Handles simple Q&A with filtering support, conversation history,
        and the ability to regenerate based on reflection feedback.
        """
        print("---NODE: SIMPLE RAG (WITH FILTERS & HISTORY)---")
        query = state["query"]
        filters = state.get("filters", {})
        messages = state.get("messages", [])
        feedback = state.get("feedback_for_improvement")

        if not query:
            print("No query provided. This should not be reached in normal flow.")
            return {"response": "I don't see a question to answer.", "sources": [], "messages": []}

        try:
            # --- 1. Retrieval Step ---
            astra_filter = {}
            title_keyword = None
            if "page_number" in filters:
                astra_filter["page_number"] = filters["page_number"]
            if "title_keyword" in filters:
                title_keyword = filters["title_keyword"]

            print(f"Retrieving context for query: '{query}' with DB filter: {astra_filter}")
            search_kwargs = {"k": 10}  # Get more docs for potential client-side filtering
            if astra_filter:
                search_kwargs["filter"] = astra_filter

            retriever = self.rag_manager.vector_store.as_retriever(search_kwargs=search_kwargs)
            docs = retriever.invoke(query)

            if title_keyword:
                docs = _filter_docs_by_title(docs, title_keyword)

            docs = docs[:5]  # Limit to top 5 after all filtering
            sources = _get_sources(docs)
            context_str = _format_docs(docs)

            # --- 2. Generation Step ---
            system_prompt = """You are a teaching assistant at Stanford in probability and statistics. You must answer the user's question based *only* on the provided context.
                Use the conversation history to understand the user's query in a broader context, but formulate the final answer based on the new context provided below.
                Crucially, you MUST use the Socratic method: do not give direct answers, but ask guiding questions to help the student think for themselves.
                If the context is insufficient, state that you cannot answer from the given information."""

            if feedback:
                print(f"  - Regenerating with feedback: {feedback}")
                # Add feedback to the system prompt for the regeneration attempt
                system_prompt += f"\n\nIMPORTANT: You are regenerating a previous response that was flawed. You MUST incorporate this feedback: '{feedback}'"

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt + "\n\nCONTEXT:\n{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])

            chain = prompt | self.reasoning_model | StrOutputParser()
            answer = chain.invoke({
                "context": context_str,
                "question": query,
                "chat_history": messages
            })

            # --- 3. Output Preparation ---
            # The AIMessage will be added to history by the reflection node if the answer is approved.
            return {
                "response": answer,
                "sources": sources,
                "feedback_for_improvement": None  # Clear feedback after use
            }

        except Exception as e:
            print(f"Error in simple RAG: {e}")
            import traceback
            traceback.print_exc()
            error_message = "I encountered an error while retrieving information. Please try rephrasing your question."
            # We still need to return a response, but it won't be added to history as it will fail reflection.
            return {
                "response": error_message,
                "sources": [],
                "feedback_for_improvement": None
            }

    def transcribe_image(self, state: GraphState) -> dict:
        """
        Node: Transcribes the user's work from an image if it exists.
        """
        print("---NODE: TRANSCRIBE IMAGE---")
        image_data = state.get("image_data")
        
        if not image_data:
            print("No image provided, but continuing critique path.")
            return {"transcribed_work": "No image was provided."}
        
        try:
            print("Transcribing image...")
            prompt = [
                {"type": "text", "text": "Transcribe the student's handwritten math work from this image."},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_data}"}
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
        """
        print("---NODE: CRITIQUE WORK---")
        query = state["query"]
        transcribed_work = state["transcribed_work"]
        
        try:
            system_prompt = """You are a Math Teaching Assistant at Stanford for Probability and Statistics. Analyze a student's question and their work.
            First, critique their approach. Second, identify the specific concept they might be struggling with.
            Third, decide if you need to look up information from the book or videos to help them.
            
            Respond in this exact JSON format:
            {{
                "critique": "A brief analysis of the student's work and approach",
                "needs_context": "A search query for the book/videos if more information is needed, otherwise an empty string"
            }}"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Student Question: {query}\n\nStudent's Transcribed Work:\n{transcribed_work}")
            ])
            
            try:
                chain = prompt | self.critique_chain
                critique_object = chain.invoke({"query": query, "transcribed_work": transcribed_work})
                critique_json = critique_object.dict()
            except Exception:
                parser = JsonOutputParser(pydantic_object=Critique)
                chain = prompt | self.reasoning_model | parser
                critique_json = chain.invoke({"query": query, "transcribed_work": transcribed_work})
        except Exception as e:
            print(f"Critique method failed: {e}")
            critique_json = {
                "critique": f"Unable to analyze the work due to a processing error. The question seems to be about: {query[:100]}...",
                "needs_context": query 
            }
        
        print(f"Critique: {critique_json}")
        return {"critique": critique_json}

    def retrieve_context(self, state: GraphState) -> dict:
        """
        Node: Retrieves context from the vector store with filtering if needed for the critique.
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
            
            if "page_number" in filters: 
                astra_filter["page_number"] = filters["page_number"]
            if "title_keyword" in filters:
                title_keyword = filters["title_keyword"]

            print(f"Retrieving context for query: '{critique_search_query}' with DB filter: {astra_filter}")
            
            search_kwargs = {"k": 6}
            if astra_filter:
                search_kwargs["filter"] = astra_filter
            
            retriever = self.rag_manager.vector_store.as_retriever(search_kwargs=search_kwargs)
            docs = retriever.invoke(critique_search_query)
            
            if title_keyword:
                docs = _filter_docs_by_title(docs, title_keyword)
            
            docs = docs[:3]
            sources = _get_sources(docs)
            return {"context": docs, "sources": sources}
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return {"context": [], "sources": []}

    def generate_socratic_response(self, state: GraphState) -> dict:
        """
        Node: Generates the final Socratic-style hint for the user, aware of chat history
        and capable of regenerating based on reflection feedback.
        """
        print("---NODE: GENERATE SOCRATIC RESPONSE---")
        query = state["query"]
        transcribed_work = state["transcribed_work"]
        critique = state.get("critique", {}).get("critique", "")
        context = state.get("context", [])
        messages = state.get("messages", [])
        context_str = _format_docs(context)
        feedback = state.get("feedback_for_improvement")

        try:
            # --- 1. Prompt Engineering Step ---
            system_prompt = """You are a helpful and encouraging AI Math Teaching Assistant at Stanford. Your role is to guide students through mathematical problems step-by-step using the Socratic method.

            Guidelines:
            - Be encouraging and supportive; acknowledge what the student did well.
            - Point out specific errors or misconceptions gently.
            - Use Steps for better clarification.
            - Ask guiding questions rather than giving direct answers.
            - Provide hints that lead students to discover solutions themselves.
            - Use context from the book or videos to support your guidance.
            - Build on the conversation history to avoid repetition.
            
            Your goal is to help students learn and understand, not just get the right answer."""

            if feedback:
                print(f"  - Regenerating with feedback: {feedback}")
                # Add feedback to the system prompt for the regeneration attempt
                system_prompt += f"\n\nIMPORTANT: You are regenerating a previous response that was flawed. You MUST incorporate this feedback: '{feedback}'"

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                # Present the user's immediate problem clearly
                ("human", "My Question: {question}\n\nMy Work:\n{work}"),
                # Provide the agent's internal monologue and context separately
                ("system", "Relevant Context from Textbook/Videos:\n{context}\n\nYour Internal Analysis:\n{critique}\n\nBased on all the information, provide a Socratic hint to the user.")
            ])

            # --- 2. Generation Step ---
            chain = prompt_template | self.reasoning_model | StrOutputParser()
            response = chain.invoke({
                "chat_history": messages,
                "question": query,
                "work": transcribed_work,
                "critique": critique,
                "context": context_str
            })
            print(f"Generated Response: {response}")

            # --- 3. Output Preparation ---
            # The AIMessage will be added to history by the reflection node if the answer is approved.
            return {
                "response": response,
                "feedback_for_improvement": None # Clear feedback after use
            }

        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
            error_message = "I encountered an error while preparing my response. Could you please try asking your question again?"
            # We still need to return a response, but it won't be added to history as it will fail reflection.
            return {
                "response": error_message,
                "feedback_for_improvement": None
            }


    def ask_clarification_question(self, state: GraphState) -> dict:
        """
        Node: Prepares the clarifying question as the final response for this turn.
        """
        print("---NODE: ASK CLARIFICATION QUESTION---")
        clarifying_question = state.get("clarifying_question", "Sorry, I'm not sure what you mean. Could you please provide more details?")
        return {
            "response": clarifying_question,
            "sources": [],
            "is_final_answer": True # Bypass reflection for this simple response
        }
    
    def reflect_on_answer(self, state: GraphState) -> dict:
        """
        Node: Critiques the generated response to ensure it meets quality standards.
        """
        print("---NODE: REFLECT ON ANSWER---")
        response_to_check = state["response"]
        
        system_prompt = """You are a Quality Control Inspector for an AI Teaching Assistant. Your task is to evaluate a generated response against strict criteria.
        
        The TA's rules are:
        1.  **Must be Socratic:** It must ask guiding questions, not give direct answers or solve the problem.
        2.  **Must be Safe and Respectful:** It must not contain any harmful, inappropriate, or off-topic content.

        Analyze the following response and determine if it follows ALL rules.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "TA's generated response to the student:\n\n---\n{response_to_check}\n\n---\n\nNow, provide your structured evaluation based on the rules.")
        ])
        
        try:
            chain = prompt | self.reflection_chain
            reflection_result = chain.invoke({"response_to_check": response_to_check})
            
            print(f"  - Reflection Result: Is Final? {reflection_result.is_final_answer}")
            if not reflection_result.is_final_answer:
                print(f"  - Feedback: {reflection_result.feedback_for_improvement}")

            return {
                "is_final_answer": reflection_result.is_final_answer,
                "feedback_for_improvement": reflection_result.feedback_for_improvement
            }
        except Exception as e:
            print(f"Error during reflection: {e}. Approving response by default.")
            # Default to approving the response if reflection fails, to avoid getting stuck
            return {"is_final_answer": True, "feedback_for_improvement": None}
        
    def finalize_response_and_update_history(self, state: GraphState) -> dict:
        """
        Node: Takes the final, approved response and adds it to the message history.
        This is the crucial step that makes the checkpointer save the AI's turn.
        """
        print("---NODE: FINALIZE AND UPDATE HISTORY---")
        
        # The reflection step has already approved this answer.
        is_final = state.get("is_final_answer", False)
        response = state.get("response")

        # Only add to history if it's a final response from a main path.
        # This prevents adding clarification questions or other intermediate steps if you don't want them.
        # In this case, we DO want to save the clarification question as the AI's turn.
        if response and is_final:
            print(f"  - Adding final AI response to history: '{response[:80]}...'")
            return {"messages": [AIMessage(content=response)]}
        
        print("  - Not a final answer, history not updated.")
        return {}

# --- EDGE FUNCTIONS ---

def decide_next_node(state: GraphState) -> str:
    """
    Determines the next step based on the routing decision.
    """
    print("---EDGE: DECIDE NEXT NODE---")
    route = state.get("route", "rag")  # Default to rag if route is not set

    if route == "clarify":
        print("Decision: Route to ask_clarification_question.")
        return "ask_clarification_question"
    elif route == "critique":
        print("Decision: Route to critique workflow.")
        return "transcribe_image"
    else: # This covers 'rag' and any other case
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

def should_end_or_regenerate(state: GraphState) -> str:
    """
    Determines whether to end the process or loop back for regeneration.
    """
    print("---EDGE: END OR REGENERATE?---")
    is_final = state.get("is_final_answer", False)
    
    if is_final:
        print("  - Decision: Answer is good. Proceeding to finalize.")
        # Return a string key that we can map to the finalize_history node
        return "finalize"
    else:
        original_route = state.get("route")
        print(f"  - Decision: Answer is flawed. Looping back to '{original_route}' path.")
        if original_route == "rag":
            return "simple_rag"
        elif original_route == "critique":
            return "generate_socratic_response"
        else:
            print("  - Warning: Unknown route for regeneration. Ending.")
            return END
        
# --- GRAPH ASSEMBLY ---

# Instantiate the agent
agent_logic = TutorAgent()

# Create the workflow graph
workflow = StateGraph(GraphState)

workflow.add_node("extract_filters", agent_logic.extract_filters)
workflow.add_node("route_question", agent_logic.route_question)
workflow.add_node("ask_clarification_question", agent_logic.ask_clarification_question)
workflow.add_node("simple_rag", agent_logic.simple_rag)
workflow.add_node("transcribe_image", agent_logic.transcribe_image)
workflow.add_node("critique_work", agent_logic.critique_work)
workflow.add_node("retrieve_context", agent_logic.retrieve_context)
workflow.add_node("generate_socratic_response", agent_logic.generate_socratic_response)
workflow.add_node("reflect_on_answer", agent_logic.reflect_on_answer) # New node
workflow.add_node("finalize_history", agent_logic.finalize_response_and_update_history)

# --- Define Edges ---

# Remove the prepare_inputs node and make extract_filters the entry point
workflow.set_entry_point("extract_filters")
workflow.add_edge("extract_filters", "route_question")
workflow.add_conditional_edges(
    "route_question",
    decide_next_node,
    {
        "ask_clarification_question": "ask_clarification_question",
        "transcribe_image": "transcribe_image",
        "simple_rag": "simple_rag",
    }
)

# Critique path
workflow.add_edge("transcribe_image", "critique_work")
workflow.add_conditional_edges("critique_work", should_retrieve_context, {
    "retrieve_context": "retrieve_context", "generate_socratic_response": "generate_socratic_response"
})
workflow.add_edge("retrieve_context", "generate_socratic_response")

# Connect response generators to reflection
workflow.add_edge("simple_rag", "reflect_on_answer")
workflow.add_edge("generate_socratic_response", "reflect_on_answer")
# Clarification questions have 'is_final_answer' set to True, so they can go to the reflection edge
workflow.add_edge("ask_clarification_question", "reflect_on_answer")

# The reflection loop logic
workflow.add_conditional_edges(
    "reflect_on_answer",
    should_end_or_regenerate,
    {
        "simple_rag": "simple_rag",
        "generate_socratic_response": "generate_socratic_response",
        # FIX: The key returned by the function is "finalize", so we map it here.
        "finalize": "finalize_history",
        # We also keep the END mapping in case the edge function returns it as a fallback.
        END: END
    }
)

# The final step is to end the graph
workflow.add_edge("finalize_history", END)

# Add memory for conversation persistence
memory = MemorySaver()

# Compile the graph
app_graph = workflow.compile(checkpointer=memory)
# app_graph = workflow.compile()
print("Combined LangGraph agent compiled successfully with memory, Astra DB compatible filtering, intelligent routing, and robust error handling.")