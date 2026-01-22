#!/usr/bin/env bash
# Exit on error
set -o errexit

# 1. Install Python Dependencies
pip install -r requirements.txt

# 2. Download and Extract FFmpeg (if not already there)
if [ ! -f ffmpeg/ffmpeg ]; then
    echo "Downloading FFmpeg..."
    # Create a directory for ffmpeg
    mkdir -p ffmpeg
    
    # Download the static build
    wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    
    # Extract it
    tar -xvf ffmpeg-release-amd64-static.tar.xz -C ffmpeg --strip-components=1
    
    # Clean up the archive
    rm ffmpeg-release-amd64-static.tar.xz
    
    echo "FFmpeg downloaded successfully."
else
    echo "FFmpeg already exists, skipping download."
fi
