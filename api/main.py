from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
from pathlib import Path
import numpy as np
try:
    from api.utils import preprocess_image, load_model
except ImportError:
    from utils import preprocess_image, load_model

app = FastAPI(
    title="Image Classification API",
    description="REST API for horse and donkey classification using CNN",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

project_root = Path(__file__).parent.parent
model_path = project_root / "models" / "horse_donkey_model.h5"
class_indices_path = project_root / "models" / "class_indices.pkl"

model = None
idx_to_class = None

def load_models():
    global model, idx_to_class
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please run training/train_model.py first."
        )
    
    if not class_indices_path.exists():
        raise FileNotFoundError(
            f"Class indices not found at {class_indices_path}. Please run training/train_model.py first."
        )
    
    model, idx_to_class = load_model(model_path, class_indices_path)
    print("Model and class indices loaded successfully!")

@app.on_event("startup")
async def startup_event():
    load_models()

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict

@app.get("/")
async def root():
    return {
        "message": "Image Classification API",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": list(idx_to_class.values()) if idx_to_class else None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    if model is None or idx_to_class is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please ensure the model files exist."
        )
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file"
            )
        
        processed_image = preprocess_image(image_bytes)
        
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = idx_to_class[predicted_class_idx]
        
        probabilities = {
            idx_to_class[i]: float(predictions[0][i])
            for i in range(len(idx_to_class))
        }
        
        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=probabilities
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

