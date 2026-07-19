from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
import json
from threading import Lock
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from openai import OpenAI

from app.core.config import get_settings
from app.services.ttl_cache import TTLCache


class EmbeddingTimeoutError(RuntimeError):
    pass


class EmbeddingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._model = None
        self._model_lock = Lock()
        self._cache = TTLCache[list[float]](ttl_seconds=self.settings.cache_ttl_seconds, max_size=512)
        self._inflight: dict[str, Future[list[float]]] = {}

    def embed(self, text: str) -> tuple[list[float], dict[str, int | bool | str]]:
        cache_key = (
            f"{self.settings.embedding_provider}:"
            f"{self.settings.embedding_model_name}:"
            f"{self.settings.runpod_endpoint_id or ''}:"
            f"{text}"
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached, {"cache_hit": True, "embedding_ms": 0, "model_ready": self._model is not None}

        started_at = perf_counter()
        future = self._inflight.get(cache_key)
        if future is None:
            future = self._executor.submit(self._embed_uncached, text)
            self._inflight[cache_key] = future

        try:
            vector = future.result(timeout=self.settings.rag_embedding_timeout_seconds)
        except TimeoutError as exc:
            raise EmbeddingTimeoutError(
                f"Embedding не успел за {self.settings.rag_embedding_timeout_seconds} сек. "
                "BGE-M3 может еще прогреваться или скачиваться; попробуй повторить запрос позже."
            ) from exc
        finally:
            if future.done():
                self._inflight.pop(cache_key, None)

        self._cache.set(cache_key, vector)
        return vector, {
            "cache_hit": False,
            "embedding_ms": int((perf_counter() - started_at) * 1000),
            "model_ready": self._model is not None,
        }

    def warmup(self) -> None:
        if self.settings.embedding_provider == "bge_m3":
            self._get_bge_model()
        elif self.settings.embedding_provider == "runpod":
            self.embed("warmup")

    def _embed_uncached(self, text: str) -> list[float]:
        if self.settings.embedding_provider == "bge_m3":
            model = self._get_bge_model()
            encoded = model.encode(
                [text],
                batch_size=1,
                max_length=2048,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            return encoded["dense_vecs"][0].astype("float32").tolist()

        if self.settings.embedding_provider == "runpod":
            return self._embed_with_runpod(text)

        if not self.settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY не задан. Для RAG-поиска через Qdrant нужен embedding API "
                "или локальный encoder с той же размерностью, что и коллекция."
            )

        client = OpenAI(api_key=self.settings.openai_api_key, timeout=self.settings.rag_embedding_timeout_seconds)
        response = client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=text,
        )
        return list(response.data[0].embedding)

    def _embed_with_runpod(self, text: str) -> list[float]:
        if not self.settings.runpod_api_key:
            raise RuntimeError("RUNPOD_API_KEY не задан. Для EMBEDDING_PROVIDER=runpod нужен ключ RunPod API.")
        if not self.settings.runpod_endpoint_id:
            raise RuntimeError("RUNPOD_ENDPOINT_ID не задан. Для EMBEDDING_PROVIDER=runpod нужен id endpoint.")

        base_url = self.settings.runpod_base_url.rstrip("/")
        endpoint_id = self.settings.runpod_endpoint_id.strip("/")
        query = urlencode({"wait": self.settings.runpod_wait_ms})
        url = f"{base_url}/{endpoint_id}/runsync?{query}"
        payload = {
            "input": {
                self.settings.runpod_embedding_input_key: text,
                "model": self.settings.embedding_model_name,
            }
        }
        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.settings.runpod_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.settings.rag_embedding_timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"RunPod embedding endpoint вернул HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"RunPod embedding endpoint недоступен: {exc.reason}") from exc

        status = str(body.get("status", "")).upper() if isinstance(body, dict) else ""
        if status and status != "COMPLETED":
            raise RuntimeError(f"RunPod embedding job завершился неуспешно: {status}")

        output = body.get("output", body) if isinstance(body, dict) else body
        return self._extract_embedding_vector(output)

    def _extract_embedding_vector(self, value: Any) -> list[float]:
        if isinstance(value, list):
            if value and all(isinstance(item, (int, float)) for item in value):
                return [float(item) for item in value]
            if value and isinstance(value[0], list):
                return self._extract_embedding_vector(value[0])

        if isinstance(value, dict):
            for key in ("embedding", "vector", "dense_vecs"):
                if key in value:
                    return self._extract_embedding_vector(value[key])
            if "embeddings" in value:
                return self._extract_embedding_vector(value["embeddings"])
            if "data" in value:
                return self._extract_embedding_vector(value["data"])
            if "output" in value:
                return self._extract_embedding_vector(value["output"])

        raise RuntimeError("RunPod embedding endpoint вернул ответ без embedding/vector.")

    def _get_bge_model(self):
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            try:
                from FlagEmbedding import BGEM3FlagModel
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "EMBEDDING_PROVIDER=bge_m3, но пакет FlagEmbedding недоступен. "
                    "Пересобери Docker image backend."
                ) from exc

            self._model = BGEM3FlagModel(self.settings.embedding_model_name, use_fp16=False)
            return self._model


embedding_service = EmbeddingService()
