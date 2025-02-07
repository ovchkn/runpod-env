#!/bin/bash

# Configuration
REGISTRY="ghcr.io"
OWNER="ovchkn"
IMAGE_NAME="runpod"
VERSION="1.0"
FULL_IMAGE_NAME="${REGISTRY}/${OWNER}/${IMAGE_NAME}:${VERSION}"

# Source API keys environment variables
if [ -f "/home/ubuntu/src/runpod-env/configs/api_keys.env" ]; then
    source /home/ubuntu/src/runpod-env/configs/api_keys.env
fi

# Check for GitHub token
if [ -z "${GITHUB_TOKEN}" ]; then
    echo "Error: GITHUB_TOKEN environment variable is not set"
    echo "Please set your GitHub Personal Access Token:"
    echo "export GITHUB_TOKEN=your_token_here"
    exit 1
fi

# Attempt to log in to GitHub Container Registry
echo "Logging in to GitHub Container Registry..."
echo "${GITHUB_TOKEN}" | docker login ghcr.io -u "${OWNER}" --password-stdin

# Check if login was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to authenticate with GitHub Container Registry"
    exit 1
fi

# Build the image
echo "Building image: ${FULL_IMAGE_NAME}"
docker build -t ${FULL_IMAGE_NAME} .

# Check if build was successful
if [ $? -ne 0 ]; then
    echo "Error: Docker build failed"
    exit 1
fi

# Push to registry
echo "Pushing image to registry..."
docker push ${FULL_IMAGE_NAME}

# Check if push was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to push image to registry"
    exit 1
fi

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