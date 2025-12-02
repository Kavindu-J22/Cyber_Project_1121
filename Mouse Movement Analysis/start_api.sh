#!/bin/bash
# Startup script for Mouse Movement Analysis API (Linux/Mac)

echo "========================================"
echo "Mouse Movement Analysis API"
echo "Zero Trust Telehealth Platform"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "Python found:"
python3 --version
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""

# Check if model checkpoint exists
if [ ! -f "checkpoints/best_model.pth" ]; then
    echo "WARNING: No trained model found at checkpoints/best_model.pth"
    echo "Please train the model first using: python train.py"
    echo ""
    echo "The API will start but may not function properly without a trained model."
    echo ""
    read -p "Press Enter to continue..."
fi

# Start the API server
echo "Starting Mouse Movement Analysis API..."
echo "API will be available at: http://localhost:8003"
echo "Press Ctrl+C to stop the server"
echo ""

python -m uvicorn src.api:app --host 0.0.0.0 --port 8003 --reload

# Deactivate virtual environment on exit
deactivate

