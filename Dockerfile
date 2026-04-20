FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/        /app/app/
COPY agent/      /app/agent/
COPY pipelines/  /app/pipelines/
COPY monitoring/ /app/monitoring/
COPY evals/      /app/evals/

RUN mkdir -p /app/prompts /tmp/logs /tmp/data

WORKDIR /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]