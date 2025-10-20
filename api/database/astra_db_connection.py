from langchain_astradb import AstraDBVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from api.config import settings
import os
from huggingface_hub import InferenceClient
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

def get_vector_store() -> AstraDBVectorStore:
    """Intializes and returns an AstraDBVectorStore instance."""

    embedding_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=settings.HF_TOKEN,
    model_name="sentence-transformers/all-MiniLM-L12-v2"  
)

    vector_store = AstraDBVectorStore(
        embedding = embedding_model,
        collection_name = settings.ASTRA_DB_COLLECTION_NAME,
        token = settings.ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint = settings.ASTRA_DB_API_ENDPOINT  ,
        namespace = settings.ASTRA_DB_KEYSPACE
    )

    return vector_store
    