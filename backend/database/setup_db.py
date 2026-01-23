# backend/database/db_setup.py

import psycopg
from psycopg import sql
from backend.config import settings
from urllib.parse import urlparse
import asyncio

async def create_database_if_missing():
    """
    Connects to the default 'postgres' database to check if the target DB exists.
    If not, it creates it.
    """
    # 1. Parse the URL to get the target DB name
    parsed_url = urlparse(settings.POSTGRES_DB_URL)
    target_db_name = parsed_url.path.lstrip('/')
    
    # 2. Create a connection URL for the 'postgres' system database
    # We replace the target DB name with 'postgres'
    system_db_url = settings.POSTGRES_DB_URL.replace(f"/{target_db_name}", "/postgres")

    print(f"DB Setup: Checking if database '{target_db_name}' exists...")

    try:
        # Try connecting to the Target DB directly.
        async with await psycopg.AsyncConnection.connect(settings.POSTGRES_DB_URL, autocommit=True):
            print(f"DB Setup: Database '{target_db_name}' already exists.")
            return
            
    except psycopg.OperationalError:
        # If connection fails, assume DB doesn't exist and try to create it
        print(f"DB Setup: Database '{target_db_name}' not found. Creating it...")
        
        try:
            # Connect to system DB 'postgres' to create the new DB
            async with await psycopg.AsyncConnection.connect(system_db_url, autocommit=True) as conn:
                async with conn.cursor() as cur:
                    # SQL injection safe method to create DB
                    await cur.execute(
                        sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target_db_name))
                    )
            print(f"DB Setup: Database '{target_db_name}' created successfully.")
            
        except Exception as e:
            print(f"DB Setup: Critical Error creating database: {e}")
            raise e