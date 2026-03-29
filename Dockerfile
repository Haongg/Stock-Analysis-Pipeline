# ============================================================
# Stage 1 – builder: install all Python dependencies
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ============================================================
# Stage 2 – runtime: lean final image
# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system dependency for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY scripts/ ./scripts/

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Default command: run the FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
