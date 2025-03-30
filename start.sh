#!/bin/bash

# Start script for GPU real-time object tracking application
# This script is intended for AWS EC2 GPU instances

echo "Starting GPU Real-time Object Tracking Application..."

# Check for CUDA
echo "Checking CUDA availability..."
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment and installing dependencies..."
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Using existing environment..."
    source venv/bin/activate
fi

# Create directories if they don't exist
mkdir -p uploads static templates

# Check if model exists, otherwise download default
if [ ! -f "models/best.pt" ]; then
    echo "Model not found, downloading default YOLOv8n model..."
    mkdir -p models
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
fi

# Start the application
echo "Starting server on port 8000..."
python main.py

echo "Server stopped." 