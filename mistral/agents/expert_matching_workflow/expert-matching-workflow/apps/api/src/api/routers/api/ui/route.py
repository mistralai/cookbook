from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(include_in_schema=False)

PAGE = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@router.get("", response_class=HTMLResponse)
async def page() -> str:
    return PAGE
