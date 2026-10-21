# Small CPU-only image for the API. The GPU work happens in the separate vLLM container.
# Build from the REPO ROOT:  docker build -f api/Dockerfile -t medscriptai-api:v1 .
# Official Python image via the AWS ECR Public mirror (avoids Docker Hub pull rate limits).
# Switch registry if one is rate-limited:  --build-arg BASE_IMAGE=python:3.12-slim
ARG BASE_IMAGE=public.ecr.aws/docker/library/python:3.12-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/common.py api/app.py ./

RUN useradd --create-home appuser
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
