# Mouse Movement Analysis - Zero Trust Telehealth Platform

## Overview

This component provides **continuous behavioral authentication** through mouse movement dynamics analysis. It is the fourth biometric modality in the Zero Trust Telehealth Platform, complementing voice, face, and keystroke dynamics to ensure unbreakable identity verification during online medical consultations.

## Key Features

- **Siamese Neural Network** with triplet loss for learning behavioral signatures
- **Real-time Continuous Verification** throughout entire consultation sessions
- **Anomaly Detection** for automated behavior, remote desktop artifacts, and user substitution
- **Adaptive Templates** that evolve with natural behavioral drift
- **REST API** for seamless integration with MERN stack frontend
- **Comprehensive Metrics** including accuracy, EER, FAR, FRR, and AUC

## Architecture

### Components

1. **Feature Extraction** (`src/mouse_preprocessing.py`)
   - Velocity, acceleration, jerk features
   - Curvature and angular variation
   - Click dynamics and trajectory analysis
   - Temporal patterns (pauses, bursts)
   - Statistical aggregations (mean, std, quartiles, skew, kurtosis)

2. **Embedding Model** (`src/mouse_embedding.py`)
   - Siamese network with shared weights
   - L2-normalized embeddings (128-dimensional)
   - Triplet loss with hard negative mining
   - Multiple similarity metrics (cosine, euclidean, manhattan)

3. **Verification System** (`src/mouse_verification.py`)
   - User enrollment with template creation
   - Single sample verification
   - Continuous verification over sequences
   - Adaptive template updating
   - Confidence scoring and thresholding

4. **Anomaly Detection** (`src/anomaly_detection.py`)
   - Isolation Forest / One-Class SVM
   - Automated behavior detection (bots, scripts)
   - Remote desktop artifact detection
   - User substitution detection
   - Risk level assessment

5. **REST API** (`src/api.py`)
   - FastAPI server with CORS support
   - Enrollment endpoint
   - Verification endpoint
   - Continuous monitoring endpoint
   - User statistics and management

## Dataset

Uses the **Balabit Mouse Dynamics Challenge** dataset:
- Training files: `Dataset/training_files/`
- Test files: `Dataset/test_files/`
- Labels: `Dataset/public_labels.csv`

Each session contains mouse events with:
- `client_timestamp`: Event timestamp
- `x`, `y`: Cursor coordinates
- `button`: Mouse button (NoButton, Left, Right)
- `state`: Event type (Move, Pressed, Released, Drag)

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)

### Setup

```bash
# Clone repository
cd "Mouse Movement Analysis"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Training

Train the Siamese network on the Balabit dataset:

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Train the model with triplet loss
- Save checkpoints to `checkpoints/`
- Generate training history plots in `logs/`

Training parameters can be configured in `config.yaml`.

### 2. Testing

Evaluate model performance:

```bash
python test.py
```

This will:
- Load the best model checkpoint
- Evaluate on test dataset
- Calculate metrics (accuracy, EER, AUC)
- Generate ROC curves and score distributions
- Save results to `logs/test_results.txt`

### 3. API Server

Start the REST API server:

```bash
# Windows:
start_api.bat

# Linux/Mac:
chmod +x start_api.sh
./start_api.sh
```

The API will be available at `http://localhost:8003`

### 4. Demo Mode

Run a quick demonstration:

```bash
python main.py --mode demo
```

## API Endpoints

### Health Check
```
GET /health
```

### Enroll User
```
POST /enroll
{
  "user_id": "doctor_123",
  "events": [
    {"timestamp": 1234567890.123, "x": 100, "y": 200, "button": "NoButton", "state": "Move"},
    ...
  ]
}
```

### Verify User
```
POST /verify
{
  "user_id": "doctor_123",
  "events": [...]
}
```

### Continuous Monitoring
```
POST /monitor
{
  "user_id": "doctor_123",
  "events": [...],
  "session_id": "consultation_456"
}
```

### Get User Statistics
```
GET /users/{user_id}/stats
```

### Delete User
```
DELETE /users/{user_id}
```

## Configuration

Edit `config.yaml` to customize:

- **Model Architecture**: Embedding dimension, hidden layers, dropout
- **Training**: Batch size, learning rate, epochs, loss function
- **Verification**: Similarity metric, thresholds (normal, alert, critical)
- **Anomaly Detection**: Detection methods, contamination rate
- **API**: Host, port, CORS settings
- **Features**: Window size, overlap, minimum events

## Performance Targets

- **Accuracy**: 85-90% user recognition
- **EER**: <15% Equal Error Rate
- **Latency**: <500ms per verification
- **False Accept Rate (FAR)**: <5%
- **False Reject Rate (FRR)**: <10%

## Integration with MERN Stack

The API is designed to integrate seamlessly with the telehealth platform:

1. **Frontend (React)**: Captures mouse events during consultation
2. **Backend (Node.js/Express)**: Forwards events to Python API
3. **MongoDB**: Stores verification logs and audit trails
4. **Zero Trust Engine**: Receives alerts for policy enforcement

## Security Features

- **Continuous Authentication**: Verifies identity throughout session, not just at login
- **Multi-threshold System**: Normal (0.70), Alert (0.60), Critical (0.50)
- **Anomaly Detection**: Detects bots, RDP, and user substitution
- **Audit Logging**: Complete forensic trail of all verifications
- **Adaptive Templates**: Prevents replay attacks through template evolution

## File Structure

```
Mouse Movement Analysis/
├── Dataset/                    # Balabit dataset
├── src/
│   ├── __init__.py
│   ├── config_loader.py       # Configuration management
│   ├── mouse_preprocessing.py # Feature extraction
│   ├── mouse_embedding.py     # Siamese network
│   ├── mouse_verification.py  # Verification logic
│   ├── anomaly_detection.py   # Anomaly detection
│   └── api.py                 # FastAPI server
├── config.yaml                # Configuration file
├── main.py                    # Entry point
├── train.py                   # Training script
├── test.py                    # Testing script
├── requirements.txt           # Dependencies
├── start_api.bat             # Windows startup
├── start_api.sh              # Linux/Mac startup
└── README.md                 # This file
```

## License

Part of the Zero Trust Telehealth Platform - Proprietary

## Contact

For questions or support, contact the development team.

