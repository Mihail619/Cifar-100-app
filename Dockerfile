FROM python:3.12-slim
# Установка системных зависимостей для OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

    
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .


CMD gunicorn src.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000