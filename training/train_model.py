import os
import numpy as np
import pickle
import json
import datetime
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def create_model(input_shape=(224, 224, 3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    project_root = Path(__file__).parent.parent
    
    data_dir = project_root / "data"
    train_dir = data_dir / "train"
    validation_dir = data_dir / "validation"
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Training data directory not found at {train_dir}")
    
    if not validation_dir.exists():
        raise FileNotFoundError(f"Validation data directory not found at {validation_dir}")
    
    print("Loading and preprocessing images...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    class_names = list(train_generator.class_indices.keys())
    print(f"Classes found: {class_names}")
    
    print("Creating model...")
    model = create_model()
    
    print("Training model...")
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        verbose=1
    )
    
    print("Evaluating model...")
    validation_loss, validation_accuracy = model.evaluate(validation_generator, verbose=0)
    
    y_true = validation_generator.classes
    y_pred_probs = model.predict(validation_generator, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n=== Model Performance ===")
    print(f"Validation Accuracy: {validation_accuracy:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("\nSaving model...")
    model_path = models_dir / "horse_donkey_model.h5"
    model.save(model_path)
    
    class_indices_path = models_dir / "class_indices.pkl"
    with open(class_indices_path, 'wb') as f:
        pickle.dump(train_generator.class_indices, f)
    
    model_info = {
        'version': '1.0.0',
        'trained_date': datetime.datetime.now().isoformat(),
        'algorithm': 'CNN_VGG16',
        'input_shape': [224, 224, 3],
        'classes': class_names,
        'class_indices': {k: int(v) for k, v in train_generator.class_indices.items()},
        'metrics': {
            'validation_accuracy': float(validation_accuracy),
            'test_accuracy': float(accuracy),
            'confusion_matrix': cm.tolist()
        }
    }
    
    metrics_path = models_dir / "model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Class indices saved to: {class_indices_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    train_model()

