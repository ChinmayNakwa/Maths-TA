from backend.database.astra_db_connection import get_vector_store
from langchain.docstore.document import Document
from typing import List

class RAGManager:
    """
    Manages the retrieval part of the RAG pipeline.
    Connects to the vector store and fetches relevant context.
    """
    def __init__(self):
        """Initializes the RAGManager by connecting to the vector store."""
        try:
            self.vector_store = get_vector_store()
            print("RAGManager: Vector store initialized successfully.")
        except Exception as e:
            print(f"RAGManager: Error initializing vector store: {e}")
            self.vector_store = None

    def retrieve_context(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieves relevant document chunks for a given query.
        
        Args:
            query: The user's question.
            k: The number of documents to retrieve.
        
        Returns:
            A list of retrieved LangChain Document objects.
        """
        if not self.vector_store:
            print("RAGManager: Cannot retrieve context, vector store not available.")
            return []
        
        print(f"RAGManager: Retrieving top {k} documents for query: '{query}'")
        try: 
            results = self.vector_store.similarity_search(query, k=k)
            print(f"RAGManager: Found {len(results)} relevant document chunks.")
            return results
        except Exception as e:
            print(f"RAGManager: An error occurred during similarity search: {e}")
            return []
        

        