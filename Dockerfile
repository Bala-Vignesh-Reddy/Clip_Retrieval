# Use slim Python image
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install only essential build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages in order
COPY requirements.txt .
RUN pip install --no-cache-dir streamlit pillow python-dotenv qdrant-client numpy matplotlib pandas scikit_learn && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers

# Final stage
FROM python:3.10-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application files
COPY testScripts/ ./testScripts/
COPY .env .

# Create necessary directories
RUN mkdir -p images testimages

# Expose port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl -f http://localhost:8501/_stcore/health

# Start command
CMD ["streamlit", "run", "testScripts/app.py", "--server.port=8501", "--server.address=0.0.0.0"]



