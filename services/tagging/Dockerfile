# Use Python 3.9 as specified by the repository's environment configuration
FROM python:3.9-slim

# Install system dependencies needed for audio processing and fetching resources
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    ffmpeg \
    libsndfile1 \
    fluid-soundfont-gm \
    build-essential \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Clone the MidiCaps repository
RUN git clone https://github.com/AMAAI-Lab/MidiCaps.git .

# I needed to downgrade the version of the music in order to be
# compatible with the python3.9
RUN sed -i s/music21==9.1.0/music21==8.3.0/g requirements.txt

# Install Python requirements
RUN pip install numpy==1.26.4
RUN pip install --no-cache-dir -r requirements.txt

#  Default command runs the feature extraction pipeline
CMD bash