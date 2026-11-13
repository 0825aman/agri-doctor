# ── Agri Doctor — IBM Cloud Code Engine Dockerfile ───────────────────────────
# Base: slim Python 3.11 for a small image footprint
FROM python:3.11-slim

# Metadata
LABEL maintainer="Agri Doctor <agridoctor@example.com>"
LABEL description="AI-powered crop disease diagnosis using IBM watsonx.ai"
LABEL version="1.0"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies (cached layer) ───────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY app.py .
COPY guardrails.py .
COPY metrics.py .
COPY dashboard.py .
# Copy Streamlit config (theme + server settings); exclude secrets.toml
COPY .streamlit/config.toml .streamlit/config.toml

# ── Streamlit telemetry opt-out ────────────────────────────────────────────────
# Prevents the interactive email prompt on first startup
RUN mkdir -p /root/.streamlit \
 && echo '[general]\nemail = ""' > /root/.streamlit/credentials.toml

# ── Runtime environment defaults ─────────────────────────────────────────────
# Sensitive values (WATSONX_API_KEY, WATSONX_PROJECT_ID) must be injected
# via IBM Code Engine secrets — never baked into the image.
ENV WATSONX_URL="https://us-south.ml.cloud.ibm.com" \
    WATSONX_VISION_MODEL="mistralai/mistral-small-3-1-24b-instruct-2503" \
    WATSONX_TEXT_MODEL="ibm/granite-4-h-small" \
    STREAMLIT_SERVER_HEADLESS="true" \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"

# ── Expose Streamlit port ─────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
