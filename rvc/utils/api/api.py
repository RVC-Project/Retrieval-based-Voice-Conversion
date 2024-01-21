from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

from rvc.utils.api.endpoints import inference

load_dotenv()

app = FastAPI()

app.include_router(inference.router)
