#!/bin/bash

echo "========================================"
echo "Voiceprint Analysis API Server"
echo "========================================"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found"
    echo "Please run: python3 -m venv venv"
    echo ""
fi

# Start the API server
echo "Starting API server..."
echo "API will be available at: http://localhost:8001"
echo "Interactive docs at: http://localhost:8001/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python main.py api

