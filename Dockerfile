# syntax=docker/dockerfile:1.3-labs

# Build a docker image for serving the Clobotics Chat Streamlit app

FROM python:3.10-slim

ARG DEBIAN_FRONTEND="noninteractive"

# Pip install first to cache the environment layer
COPY requirements.txt /app/

RUN <<EOF
set -e
# apt-get update
# apt-get install -y --no-install-recommends build-essential
# rm -rf /var/lib/apt/lists/*
# apt-get clean
python3 -m pip install --no-cache-dir -r /app/requirements.txt
python3 -m pip cache purge
EOF

# Copy the code layer
ADD __app.tar.gz /app
# # Add secrets file
# ADD .streamlit/secrets.toml /app/.streamlit
WORKDIR /app
