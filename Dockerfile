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

COPY ./pyproject.toml .
COPY ./setup.cfg .
COPY ./src ./src

RUN pip install --no-cache-dir .

ENV APBS_BIN_DIR="/opt/apbs/bin"
ENV PATH="${APBS_BIN_DIR}:${PATH}"
ENV APBS_PATH="${APBS_BIN_DIR}/apbs"
ENV PYTHON="/usr/bin/python3"

RUN useradd -ms /bin/bash appuser
USER appuser
