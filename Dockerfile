FROM python:3.12-slim

# System deps for scikit-learn / xgboost native builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot
COPY accum_bot.py .

# Ephemeral local storage — Supabase is the real persistence layer
ENV PERSIST_DIR=/tmp/accum_botdata

# Health server port (Railway injects PORT automatically)
EXPOSE 8080

CMD ["python", "-u", "accum_bot.py"]
