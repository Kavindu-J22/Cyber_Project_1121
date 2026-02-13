#!/bin/bash
# Start Face Verification API Server
# Zero Trust Telehealth Platform

echo "========================================"
echo "Face Verification API - Starting"
echo "========================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

# Check if requirements are installed
python3 -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Check if model file exists
if [ ! -f "models/best_resnet50_triplet.pth" ]; then
    echo ""
    echo "WARNING: Model file not found!"
    echo "Expected location: models/best_resnet50_triplet.pth"
    echo ""
    echo "Please place your trained model file in the models folder."
    echo ""
    exit 1
fi

echo "Starting Face Verification API on port 8004..."
echo ""

python3 main.py api
