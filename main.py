from dotenv import load_dotenv
load_dotenv()

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
from requests.exceptions import ConnectionError
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
from schemas.message import Message
from schemas.response import ChatResponse
from logger import get_logger
from db_agent import DatabaseAgent

logger = get_logger(__name__)
llm = init_chat_model(model="gemini-1.5-pro", model_provider="google_genai")
supported_database_names = ["cred"]
checkpointer = InMemorySaver()

# Track how many messages have been processed per thread to avoid duplicate file notifications
processed_message_counts: dict[str, int] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(periodic_cleanup())
    yield

app = FastAPI(title="Wizard chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # or ["*"] for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

export_directory = Path("exports")
export_directory.mkdir(exist_ok=True)
app.mount("/downloads", StaticFiles(directory=str(export_directory)), name="downloads")

# Background task to clean up old files
async def cleanup_old_files():
    """Remove files older than 24 hours."""
    try:
        cutoff_time = datetime.now() - timedelta(hours=24)
        for file_path in export_directory.glob("*"):
            if file_path.is_file():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff_time:
                    file_path.unlink()
                    logger.info(f"Cleaned up old file: {file_path.name}")
    except Exception as e:
        logger.error(f"Error during file cleanup: {e}")

async def periodic_cleanup():
    """Run file cleanup every hour."""
    while True:
        await asyncio.sleep(3600) # Wait 1 hour
        await cleanup_old_files()

@app.post("/chat/v1")
async def chat_v1(message: Message, db_name: str, thread_id: str):
    if db_name not in supported_database_names:
        raise NotImplementedError(f"This database is not supported yet: {db_name}")
        
    db_host = os.environ.get(f"{db_name}_db_host") # type: ignore
    db_port = os.environ.get(f"{db_name}_db_port") # type: ignore
    db_user = os.environ.get(f"{db_name}_db_user") # type: ignore
    db_password = os.environ.get(f"{db_name}_db_password", "") # type: ignore
    db_name = os.environ.get(f"{db_name}_db_name") # type: ignore

    if not db_host or not db_port or not db_user or not db_name:
        raise NotImplementedError(f"This database is not supported yet: {db_name}")
    
    db = SQLDatabase.from_uri(f"clickhouse+http://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
    agent = DatabaseAgent(llm=llm, db=db, checkpointer=checkpointer, export_directory="exports")
    
    try:
        await agent.connect_to_mcp_server()
        
        result = await agent.run(message.content, processed_message_counts, thread_id)
        
        # Parse created files to extract download URLs
        download_links = []
        for file_info in result.get("created_files", []):
            if "Download:" in file_info:
                # Extract the download URL
                url_part = file_info.split("Download:")[-1].strip().rstrip(")")
                if url_part:
                    base_url = os.environ.get("SERVER_URL", "http://localhost:8000")
                    full_url = f"{base_url}{url_part}"
                    download_links.append(full_url)
        
        # Enhance response with download links if any
        response_text = result["response"]
        if download_links:
            response_text += "\n\n🔗 **Download Links:**\n"
            for i, link in enumerate(download_links, 1):
                filename = os.path.basename(link)
                response_text += f"{i}. [{filename}]({link})\n"
        
        return ChatResponse(
            response=response_text,
            created_files=download_links,
            thread_id=thread_id,
            success=True,
            error=result.get("error")
        )
        
    except ConnectionError as e:
        logger.error(f"Failed to connect: \n{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing request: \n{e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await agent.close_mcp_server()