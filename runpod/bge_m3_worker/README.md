# BGE-M3 RunPod Serverless Worker

Standalone RunPod Serverless handler for `BAAI/bge-m3` embeddings.

## RunPod Build Settings

- Dockerfile path: `runpod/bge_m3_worker/Dockerfile`
- Build context: `runpod/bge_m3_worker`

## Endpoint Payload

```json
{
  "input": {
    "prompt": "товар пришел сломанный"
  }
}
```

`text` is also supported as a fallback:

```json
{
  "input": {
    "text": "товар пришел сломанный"
  }
}
```

Response:

```json
{
  "embedding": [0.1, 0.2],
  "dim": 1024,
  "model": "BAAI/bge-m3"
}
```

## Main Backend Env

Set these variables for the existing backend:

```bash
export EMBEDDING_PROVIDER=runpod
export RUNPOD_ENDPOINT_ID=...
export RUNPOD_API_KEY=...
export RUNPOD_EMBEDDING_INPUT_KEY=prompt
```

Do not commit `RUNPOD_API_KEY` or any other secret to the repository.
