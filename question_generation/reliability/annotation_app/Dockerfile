FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static
COPY data/questions_2015.json ./data/questions_2015.json
COPY secrets.example.json ./secrets.json

ENV PYTHONUNBUFFERED=1

# Render sets $PORT
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8765}
