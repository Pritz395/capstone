FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY app.py .
COPY src ./src
COPY templates ./templates
COPY models ./models
COPY artifacts ./artifacts
COPY data ./data

ENV PORT=8000
EXPOSE 8000

CMD gunicorn -b 0.0.0.0:${PORT} app:app --workers 1 --threads 4 --timeout 120
