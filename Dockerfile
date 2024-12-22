# Stage 1: Builder
FROM python:3.9-slim as builder

# Set environment variables to prevent Python from writing .pyc files and to buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container to /app
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

# (Optional) Install Poetry for dependency management
# RUN pip install poetry

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Create a virtual environment
RUN python -m venv /opt/venv

# Activate the virtual environment and update PATH
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 2: Final runtime image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# Set the working directory in the container to /app
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application code into the container
COPY . /app

# Command to run the Chainlit server using shell form for environment variable expansion
CMD python -m chainlit run app.py -h 0.0.0.0 --port ${PORT}
