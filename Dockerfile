# Use the official Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

# Install git and other necessary system dependencies
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Set environment variables
ENV UV_SYSTEM_PYTHON=1
ENV PATH="/root/.local/bin:$PATH"

# Copy pyproject.toml and uv.lock* first to leverage Docker cache
COPY pyproject.toml uv.lock* ./

# Install dependencies using pyproject.toml and lockfile if it exists
RUN uv pip sync pyproject.toml

# Copy the rest of the application code
COPY . .

# Run Chainlit app
CMD uv run chainlit run app.py -h --host 0.0.0.0 --port ${PORT}