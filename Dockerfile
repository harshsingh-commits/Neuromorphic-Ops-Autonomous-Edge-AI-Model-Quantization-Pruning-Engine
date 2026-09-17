FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app


# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libxrender1 \
       curl \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# PYTHON DEPENDENCIES
# ============================================================

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt


# ============================================================
# APPLICATION
# ============================================================

COPY . .


# ============================================================
# RUNTIME DIRECTORIES
# ============================================================

RUN mkdir -p \
    /app/models \
    /app/outputs \
    /app/reports \
    /app/runs \
    /app/data


# ============================================================
# NON-ROOT USER
# ============================================================

RUN useradd \
    --create-home \
    --uid 10001 \
    --shell /usr/sbin/nologin \
    appuser \
    && chown -R appuser:appuser /app


USER appuser


# ============================================================
# NETWORK
# ============================================================

EXPOSE 8000
EXPOSE 8501


# ============================================================
# CONTAINER HEALTHCHECK
# ============================================================

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=20s \
    --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1


# ============================================================
# START APPLICATION
# ============================================================

# ============================================================
# START APPLICATION
# ============================================================

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]