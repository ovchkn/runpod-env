#!/bin/bash

# Configuration
REGISTRY="ghcr.io"
OWNER="ovchkn"
IMAGE_NAME="runpod"
VERSION="1.0"
FULL_IMAGE_NAME="${REGISTRY}/${OWNER}/${IMAGE_NAME}:${VERSION}"

# Source API keys environment variables
source ../configs/api_keys.env

# Ensure we're logged into GitHub Container Registry
if ! docker info | grep -q "ghcr.io"; then
    echo "Please login to GitHub Container Registry first:"
    echo "echo \$GITHUB_TOKEN | docker login ghcr.io -u ${OWNER} --password-stdin"
    exit 1
fi

# Build the image
echo "Building image: ${FULL_IMAGE_NAME}"
docker build -t ${FULL_IMAGE_NAME} .

# Push to registry
echo "Pushing image to registry..."
docker push ${FULL_IMAGE_NAME}

echo "Deployment complete! Image is available at: ${FULL_IMAGE_NAME}"
echo ""
echo "To run the container:"
echo "docker run -d \\"
echo "  --name runpod-env \\"
echo "  -v /path/to/network/storage:/network/mlops \\"
echo "  -e EXTERNAL_HOST=your.host.com \\"
echo "  -e EXTERNAL_PORT=11434 \\"
echo "  -p 11434:11434 \\"
echo "  -p 5000:5000 \\"
echo "  -p 8000:8000 \\"
echo "  -p 8888:8888 \\"
echo "  -p 3000:3000 \\"
echo "  -p 6006:6006 \\"
echo "  --gpus all \\"
echo "  ${FULL_IMAGE_NAME}"