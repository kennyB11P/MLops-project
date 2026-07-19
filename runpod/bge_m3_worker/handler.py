from __future__ import annotations

import os
import sys
import traceback
from typing import Any

import runpod


MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
MAX_LENGTH = int(os.getenv("BGE_MAX_LENGTH", "2048"))
BATCH_SIZE = int(os.getenv("BGE_BATCH_SIZE", "1"))
_MODEL: Any | None = None
_MODEL_LOAD_ERROR: str | None = None


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_model() -> Any:
    global _MODEL, _MODEL_LOAD_ERROR

    if _MODEL is not None:
        return _MODEL
    if _MODEL_LOAD_ERROR is not None:
        raise RuntimeError(_MODEL_LOAD_ERROR)

    try:
        import torch
        from FlagEmbedding import BGEM3FlagModel

        use_fp16 = _bool_env("BGE_USE_FP16", torch.cuda.is_available())
        _MODEL = BGEM3FlagModel(MODEL_NAME, use_fp16=use_fp16)
        return _MODEL
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        _MODEL_LOAD_ERROR = f"Failed to load embedding model {MODEL_NAME}: {exc}"
        raise RuntimeError(_MODEL_LOAD_ERROR) from exc


def _input_text(event: dict[str, Any]) -> str:
    payload = event.get("input") or {}
    if not isinstance(payload, dict):
        raise ValueError("event.input must be an object")

    value = payload.get("prompt")
    if value is None:
        value = payload.get("text")
    if value is None:
        raise ValueError('event.input.prompt is required; "text" is supported as fallback')

    text = str(value).strip()
    if not text:
        raise ValueError("prompt must not be empty")
    return text


def handler(event: dict[str, Any]) -> dict[str, Any]:
    try:
        text = _input_text(event)
        model = get_model()
        encoded = model.encode(
            [text],
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            normalize_embeddings=True,
        )
        vector = encoded["dense_vecs"][0].astype("float32").tolist()
    except Exception as exc:  # noqa: BLE001
        return {
            "error": str(exc),
            "model": MODEL_NAME,
            "dim": 0,
        }

    return {
        "embedding": vector,
        "dim": len(vector),
        "model": MODEL_NAME,
    }


runpod.serverless.start({"handler": handler})
