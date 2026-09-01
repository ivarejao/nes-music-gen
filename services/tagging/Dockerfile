# Use Python 3.9 as specified by the repository's environment configuration
FROM python:3.9-slim

# Install system dependencies needed for audio processing and fetching resources
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

COPY requirements.txt requirements.txt

# Install numpy in order to be compatible with vamp installation
RUN pip install numpy==1.26.4
RUN pip install --no-cache-dir -r requirements.txt

#  Default command runs the feature extraction pipeline
CMD python3 generate_tags.py --dataset-path /app/dataset/