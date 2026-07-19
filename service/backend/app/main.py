from threading import Thread

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.services.embedding_service import embedding_service
from app.ui import router as ui_router

settings = get_settings()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для MVP. В production указать домен сайта.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}


@app.on_event("startup")
def warmup_embedding_model() -> None:
    if not settings.rag_warmup_on_startup:
        return

    Thread(target=embedding_service.warmup, daemon=True).start()


app.include_router(ui_router)
app.include_router(api_router, prefix=settings.api_prefix)
