#!/bin/bash

set -e

IMAGE_NAME=$1

# If tag is not empty, then use it as the tag

if [ -z "$IMAGE_NAME" ]; then
    IMAGE_NAME="tipani86/xiaopan-ai:uk_frs"
fi

git archive -v -o __app.tar.gz --format=tar.gz HEAD
docker build -t $IMAGE_NAME -f Dockerfile .
