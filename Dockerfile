# Vanara.ai backend — production image
# Pure-Python deps only (xhtml2pdf + PyPDF2 for PDF); no native libs needed.

FROM python:3.14-slim AS base

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app/ ./app/
COPY templates/ ./templates/

# Non-root user for security
RUN useradd --create-home --shell /bin/bash vanara && chown -R vanara:vanara /app
USER vanara

ENV PORT=8000
EXPOSE 8000

# Healthcheck uses the /health endpoint (open, no auth)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen(\"http://localhost:${PORT}/health\", timeout=3).status == 200 else 1)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]