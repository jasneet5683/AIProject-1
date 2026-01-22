#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status.
set -o errexit

echo "Build started..."

# 1. Install Python Dependencies
pip install -r requirements.txt

# 2. Download and Install FFmpeg (Static Build)
if [ ! -f ffmpeg/ffmpeg ]; then
    echo "Downloading FFmpeg..."
    mkdir -p ffmpeg
    # Download the static build
    wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    # Extract it
    tar -xvf ffmpeg-release-amd64-static.tar.xz -C ffmpeg --strip-components=1
    # Cleanup compressed file
    rm ffmpeg-release-amd64-static.tar.xz
    echo "FFmpeg downloaded successfully."
else
    echo "FFmpeg directory already exists, skipping download."
fi

echo "Build completed successfully."
