# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by OpenCV and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
# Pinning pip version for consistency
RUN pip install --no-cache-dir --upgrade pip==23.3.1

# Install PyTorch with CUDA 11.3 support
RUN pip install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other dependencies from requirements.txt and specific packages for RunPod
RUN pip install --no-cache-dir -r requirements.txt runpod yt-dlp gofile Flask Werkzeug==2.0.3

# Copy the rest of the application code into the container
COPY . .

# Download models and other necessary files
# Ensure the download.sh script is executable and handles paths correctly within the Docker image
RUN chmod +x download.sh \
    && ./download.sh

# Ensure the config.ini path is correct or provide a default one if necessary
# If config.ini is not in the root, adjust the COPY command or create a default one.
# Assuming config/config.ini is the correct path from the repo root.
RUN if [ ! -f config/config.ini ]; then \
        echo "Warning: config/config.ini not found. Creating a default or placeholder if necessary."; \
        mkdir -p config; \
        echo "[DEFAULT]" > config/config.ini; \
        echo "temp_dir = /app/temp_data" >> config/config.ini; \
        echo "result_dir = /app/results" >> config/config.ini; \
        echo "watermark_path = /app/path/to/your/watermark.png" >> config/config.ini; \
        echo "digital_auth_path = /app/path/to/your/digital_auth.png" >> config/config.ini; \
    fi

# Create necessary directories that might be expected by the application at runtime
RUN mkdir -p /app/temp_data /app/results

# Command to run the application using RunPod's Python SDK
# The -u flag ensures that Python output is sent straight to stdout/stderr without being buffered
CMD ["python", "-u", "runpod_handler.py"]
