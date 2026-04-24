# Deployment Guide

## Local Deployment

1. Install dependencies: `pip install -r requirements.txt`
2. Run the web app: `python src/app.py`
3. Open the local URL in your browser

## Hugging Face Spaces Deployment

1. Create a new Space on Hugging Face
2. Upload the following files:
   - `src/app.py`
   - `src/inference.py`
   - `src/items.py`
   - `requirements.txt`
   - Optional: `models/gpt2-pricer/` if you have a fine-tuned model
3. Set the main file to `src/app.py`
4. Deploy!

> Note: If no local fine-tuned model is present, the app will fall back to the base `gpt2` model and still start successfully.

## Docker Deployment

Create a Dockerfile:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 7860

CMD ["python", "src/app.py"]
```

Build and run:
```bash
docker build -t product-pricer .
docker run -p 7860:7860 product-pricer
```