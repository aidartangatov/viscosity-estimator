FROM ubuntu:22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    unzip \
    wget \
    libarpack2-dev \
    libf2c2-dev \
    libeigen3-dev \
    libboost-dev \
    python3-pip \
    libopenblas-serial-dev \
    liblapack-dev \
    libsuitesparse-dev \
    libsuperlu-dev


RUN wget https://github.com/Electrostatics/apbs/releases/download/v3.4.1/APBS-3.4.1.Linux.zip && \
    unzip APBS-3.4.1.Linux.zip && \
    rm APBS-3.4.1.Linux.zip && \
    mv APBS-3.4.1.Linux /opt/apbs


WORKDIR /app

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./pyproject.toml .
COPY ./setup.cfg .
COPY ./setup.py .
COPY ./MANIFEST.in .
COPY ./src ./src

ENV PYTHONPATH /app/src

ENV APBS_BIN_DIR="/opt/apbs/bin"
ENV PATH="${APBS_BIN_DIR}:${PATH}"
ENV APBS="${APBS_BIN_DIR}/apbs"
ENV PYTHON="/usr/bin/python3"
