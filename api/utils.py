import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

def load_model(model_path, class_indices_path):
    model = keras.models.load_model(model_path)
    
    import pickle
    with open(class_indices_path, 'rb') as f:
        class_indices = pickle.load(f)
    
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    return model, idx_to_class

