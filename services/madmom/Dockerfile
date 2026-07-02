FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    gcc g++ \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# madmom is compatible with Python 3.6–3.9
RUN pip install --upgrade pip
RUN pip install \
cython>=0.25 \
mido>=1.2.6 \
"numpy>=1.13.4,<1.20.0" \
scipy>=0.1

#RUN pip install \
#	cython \
#	scipy \
#	numpy \ 
#	mido
RUN pip install --no-build-isolation madmom

RUN pip install fastapi uvicorn python-multipart

# Verify the CLI tool was installed
RUN which TempoDetector
