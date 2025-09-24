# syntax=docker/dockerfile:1.6

# Minimal Python base
FROM python:3.11-slim

# System deps + AWS CLI v2 + uv
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl unzip ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip && \
    unzip awscliv2.zip && ./aws/install -i /usr/local/aws -b /usr/local/bin && \
    rm -rf awscliv2.zip aws && \
    curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --install-dir /usr/local/bin

# App dir
WORKDIR /app

# (1) Create fresh venv with uv
# (2) Install deps from requirements.txt with uv
# Use BuildKit cache for faster rebuilds and avoid hard-links on overlayfs
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"
ENV UV_LINK_MODE=copy
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy only requirements first to leverage layer caching
COPY requirements.txt /app/requirements.txt
RUN uv venv "${VIRTUAL_ENV}" -p 3.11 && \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r /app/requirements.txt

# Then copy the rest of the project
COPY . /app

# Optional: show versions at build (useful for debug)
RUN python -V && uv --version && aws --version

# Default command (adjust to your entrypoint)
# CMD ["python", "-m", "your_module"]