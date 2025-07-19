from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any

from backend.database.astra_db_connection import get_vector_store
from backend.core.schemas import SourceDocument
from backend.config import settings
from langchain.docstore.document import Document

def _format_docs(docs: List[Document]) -> str:
    """
    Prepares the retrieved documents for insertion into the prompt.
    """
    formatted_docs = []
    for i, doc in enumerate(docs):
        source_type = doc.metadata.get('source', 'unknow')
        location = ""
        if source_type == 'PROBABILITY_FOR_COMPUTER_SCIENTIST':
            location = f"Page {doc.metadata.get('page_number', 'N/A')}"
        elif source_type == 'VIDEO':
            title = doc.metadata.get('title', 'N/A')
            ss = int(doc.metadata.get('title','N/A'))
            h, m = divmod(ss, 3600)
            m, s = divmod(m, 60)
            timestamp = f"{h:02d}:{m:02d}:{s:02d}"
            location = f"Video '{title}' at {timestamp}"

        doc_string = f"--- Context Snippet {i+1} from {source_type} ({location}) ---\n"
        doc_string += doc.page_content
        formatted_docs.append(doc_string)

    return "\n\n".join(formatted_docs)

def _get_sources(docs: List[Document]) -> List[SourceDocument]:
    """
    Creates a list of SourceDocument objects from the retrieved docs.
    """
    sources = []
    unique_sources = set()

    for doc in docs:
        source_type = doc.metadata.get('source', 'unkown')
        url = doc.metadata.get("video_url") if source_type == 'video' else doc.metadata.get("image_path")

        source_key = ""
        if source_type == 'PROBABILITY_FOR_COMPUTER_SCIENTIST':
            location = f"Page {doc.metadata.get('page_number', 'N/A')}"
            source_key = f"book_page_{doc.metadata.get('page_number')}"
        elif source_type == 'video':
            title = doc.metadata.get('title', 'N/A')
            ss = int(doc.metadata.get('start_time_sec', 0))
            h, m = divmod(ss, 3600); m, s = divmod(m, 60)
            timestamp = f"{h:02d}:{m:02d}:{s:02d}"
            location = f"'{title}' at {timestamp}"
            source_key = f"video_{doc.metadata.get('video_id')}" # Group by video

        if source_key not in unique_sources:
            sources.append(SourceDocument(source_type=source_type, location=location, url=url))
            unique_sources.add(source_key)
            
    return sources

def get_rag_chain():
    """
    Builds and returns a LangChain RAG chain.
    """
    # 1. Get the retriever from our AstraDB vector store
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 2. Define the prompt template
    template = """
    You are an teaching assistant at Stanford in probability and statistics. You must answer the user's question based *only* on the provided context.
    The context contains snippets from a textbook and video lectures. Synthesize the information from all context snippets to provide a comprehensive and clear answer.
    If the context is insufficient, state that you cannot answer from the given information.

    CONTEXT:
    {context}

    USER QUESTION:
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 3. Define the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, api_key = settings.GOOGLE_API_KEY)

    # 4. Construct the RAG chain using LCEL
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: _format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(
        answer=rag_chain_from_docs,
        sources=(lambda x: _get_sources(x["context"]))
    )
    
    return rag_chain_with_source

    