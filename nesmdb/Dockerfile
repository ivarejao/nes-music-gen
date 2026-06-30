# Step 1: Use a reliable Python 2.7 image
FROM python:2.7-stretch

# Step 2: The Archive Fix
RUN rm -rf /etc/apt/sources.list.d/* && \
    echo "deb http://archive.debian.org/debian/ stretch main" > /etc/apt/sources.list && \
    echo "deb http://archive.debian.org/debian-security stretch/updates main" >> /etc/apt/sources.list && \
    echo "Acquire::Check-Valid-Until false;" > /etc/apt/apt.conf.d/10no--check-valid-until

# Step 3: Install system dependencies 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Set up the working directory
WORKDIR /app

# Step 5: Fix outdated pip, setuptools, and wheel versions for Python 2.7
RUN pip install --no-cache-dir --upgrade "pip<21.0" "setuptools<45.0" "wheel<0.38.0"

# Step 6: Clone the repository
RUN git clone https://github.com/chrisdonahue/nesmdb.git .

# Step 7: Patch setup.py to handle the VGMPlay GitHub repo rename
RUN sed -i 's/vgmplay-0.40.8/vgmplay-legacy-0.40.8/g' setup.py

# Step 8: Install the requirements individually
RUN pip install --no-cache-dir "numpy>=1.7.0,<1.17.0" "scipy>=1.0.0,<1.3.0"
RUN pip install --no-cache-dir "Pillow>=5.1.0,<7.0.0" "tqdm>=4.19.9"

RUN pip install --no-cache-dir "llvmlite==0.31.0" "numba==0.47.0" "resampy==0.2.2"

RUN pip install --no-cache-dir "nose==1.3.7" "mido>=1.1.16" "pretty_midi==0.2.8"
RUN pip install --no-cache-dir "librosa>=0.6.1,<0.8.0"

# Install requirements for the API wraper
# TODO move copy to begin of the file
COPY requirements.txt requirements.txt 
RUN pip install -r requirements.txt

# Step 9: Finalize installation (note the .py!)
RUN python setup.py install

# rename source directory in order to enviroment pick the installed pack and not the 
# directory structure
RUN mv nesmdb nesmdb_src

CMD python app.py
