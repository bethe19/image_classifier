# Image Classifier

A deep learning project for classifying images using neural networks.

## Description

This project implements an image classification model that can train on image datasets and make predictions on new images.

## How to Run

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/bethe19/image_classifier.git
cd image_classifier
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
- On Windows:
```bash
.venv\Scripts\activate
```
- On macOS/Linux:
```bash
source .venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. To start the API server:
```bash
python run_api.py
```

The API will be available at `http://localhost:5000`

2. To access the frontend:
- Navigate to the `frontend` directory in your browser

### Training the Model

1. Place your training images in the `training` directory
2. Run the training script from the `models` directory
3. Trained models will be saved in the `models` directory

## Project Structure

- `api/` - Flask API for serving predictions
- `frontend/` - Web interface
- `models/` - Trained model files
- `training/` - Training data and scripts
- `run_api.py` - Entry point for the API server
