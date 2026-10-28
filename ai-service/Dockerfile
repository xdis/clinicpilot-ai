FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
ENV CHECKPOINT_DB=/data/checkpoints.sqlite3 MEMORY_DB=/data/memory.sqlite3 HF_HOME=/data/hf HOME=/data
# non-root user; /data is the volume for checkpoints, long-term memory and the Whisper model cache (HF_HOME)
RUN useradd --system --uid 10001 app && mkdir -p /data && chown app /data
USER 10001
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
