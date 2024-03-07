import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from rvc.wrapper.api.endpoints import inference

load_dotenv()

app = FastAPI()

app.include_router(inference.router)
