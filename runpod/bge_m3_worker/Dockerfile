FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf_cache \
    TRANSFORMERS_CACHE=/runpod-volume/hf_cache

COPY runpod/bge_m3_worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY runpod/bge_m3_worker/handler.py .

CMD ["python", "-u", "handler.py"]
