FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    libfontconfig1 \
    libfreetype6 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libxext6 \
    libxrender1 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/autocleaneeg-icvision
