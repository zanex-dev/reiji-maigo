# Optional container image for MCP registries/sandboxes (Glama, Smithery, Docker MCP).
# Local installs require zero Docker or third-party packages: pip install -e . --no-deps
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    REIJI_MAIGO_HOME=/tmp/reiji_maigo_data

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY reiji_maigo ./reiji_maigo

RUN pip install --no-cache-dir --no-deps .

ENTRYPOINT ["reiji", "mcp"]
