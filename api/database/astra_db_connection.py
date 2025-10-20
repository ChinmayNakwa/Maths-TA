from langchain_astradb import AstraDBVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.config import settings

def get_vector_store() -> AstraDBVectorStore:
    """Intializes and returns an AstraDBVectorStore instance."""

    embedding_model = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL
    )

    vector_store = AstraDBVectorStore(
        embedding = embedding_model,
        collection_name = settings.ASTRA_DB_COLLECTION_NAME,
        token = settings.ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint = settings.ASTRA_DB_API_ENDPOINT  ,
        namespace = settings.ASTRA_DB_KEYSPACE
    )

    return vector_store
    