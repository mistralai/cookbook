from db import dispose_engine
from fastapi import FastAPI


async def shutdown(_app: FastAPI) -> None:
    await dispose_engine()
