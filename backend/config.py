from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Settings(BaseSettings):
    GOOGLE_API_KEY: str = Field(..., env="GOOGLE_API_KEY")

    ASTRA_DB_API_ENDPOINT: str = Field(..., env="ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_APPLICATION_TOKEN: str = Field(..., env="ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_KEYSPACE: str = Field(..., env="ASTRA_DB_KEYSPACE") 
    ASTRA_DB_COLLECTION_NAME: str = Field(..., env="ASTRA_DB_COLLECTION_NAME")

    CLOUDINARY_CLOUD_NAME: str = Field(..., env="CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY: str = Field(..., env="CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET: str = Field(..., env="CLOUDINARY_API_SECRET")

    EMBEDDING_MODEL: str = Field("all-MiniLM-L12-v2")
    EMBEDDING_DIMENSION: int = 384

    CHUNK_SIZE: int = 1000         
    CHUNK_OVERLAP: int = 150
    PDF_PATH: str = r"C:\Users\chinm\Desktop\GitHub\Maths-TA\ProbabilityForComputerScientists.pdf"
    #Field(..., env="PROBABILITY_FOR_COMPUTER_SCIENTIST_PATH")
    YOUTUBE_PLAYLIST_URL: str = "https://www.youtube.com/playlist?list=PLoROMvodv4rOpr_A7B9SriE_iZmkanvUg"
    #Field(..., env="CS_109_PLAYLIST_URL") 
    IMAGE_OUTPUT_DIR: str = "output_images"
    

    class Config:
        extra = "allow"
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()     