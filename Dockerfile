FROM python:3.13-slim

# Install poppler for PDF rendering (pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (cache layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" || pip install --no-cache-dir .

# Copy application code
COPY retirement_planner/ retirement_planner/
COPY static/ static/
COPY tests/ tests/

# Create sessions directory
RUN mkdir -p /root/.retirement_planner/sessions

EXPOSE 8000

CMD ["uvicorn", "retirement_planner.web:app", "--host", "0.0.0.0", "--port", "8000"]
