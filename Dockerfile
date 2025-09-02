FROM python:3.10-slim

# Env agar ringan & headless
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# System deps (kalau perlu font untuk matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway menyediakan $PORT
EXPOSE 8080
CMD gunicorn -b 0.0.0.0:$PORT app:app
