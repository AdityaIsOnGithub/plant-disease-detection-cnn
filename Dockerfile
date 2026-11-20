FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Increase pip timeout to 1000 seconds and add retries so large downloads like PyTorch don't timeout
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]