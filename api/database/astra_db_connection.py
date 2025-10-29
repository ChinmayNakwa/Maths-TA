from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from api.config import settings

def get_vector_store() -> AstraDBVectorStore:
    """Initializes and returns an AstraDBVectorStore instance."""

    embedding_model = HuggingFaceEndpointEmbeddings(
        model=settings.EMBEDDING_MODEL,
        huggingfacehub_api_token=settings.HF_TOKEN
    )

    vector_store = AstraDBVectorStore(
        embedding=embedding_model,
        collection_name=settings.ASTRA_DB_COLLECTION_NAME,
        token=settings.ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=settings.ASTRA_DB_API_ENDPOINT,
        namespace=settings.ASTRA_DB_KEYSPACE
    )

    return vector_store