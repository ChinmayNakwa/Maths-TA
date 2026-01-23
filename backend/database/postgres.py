# backend/database/postgres.py

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from backend.config import settings

# Connection configuration
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

# Initialize the pool with open=False to prevent warnings.
# We will explicitly open it in app.py
pool = AsyncConnectionPool(
    conninfo=settings.POSTGRES_DB_URL,
    max_size=20,
    kwargs=connection_kwargs,
    open=False 
)

# Initialize the checkpointer with the pool
checkpointer = AsyncPostgresSaver(pool)

async def delete_thread_data(thread_id: str):
    """
    Deletes all checkpoint data associated with a specific thread_id.
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # Delete from the main checkpoints table
            await cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = %s", 
                (thread_id,)
            )
            # Delete from the writes/blobs table
            await cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = %s", 
                (thread_id,)
            )