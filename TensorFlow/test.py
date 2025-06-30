import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
from datetime import datetime

class VitalSignsExtractor:
    def __init__(self):
        self.model = None
        self.image_size = (224, 224)
        self.vital_signs_labels = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'spo2', 'temperature', 'pulse_rate']
        
    def preprocess_image(self, image_path):
        """Preprocess image for the model"""
        # Read and resize image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, self.image_size)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def load_dataset(self, dataset_dir="generated_heart_monitors"):
        """Load and prepare the dataset"""
        # Load metadata
        metadata_path = os.path.join(dataset_dir, "dataset_info.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            dataset_info = json.load(f)
        
        images = []
        labels = []
        
        print(f"Loading {len(dataset_info)} images...")
        
        for item in dataset_info:
            image_path = os.path.join(dataset_dir, item['filename'])
            if os.path.exists(image_path):
                try:
                    # Load and preprocess image
                    image = self.preprocess_image(image_path)
                    images.append(image)
                    
                    # Extract vital signs as labels
                    vitals = item['vitals']
                    label_vector = [
                        vitals['heart_rate'],
                        vitals['systolic_bp'],
                        vitals['diastolic_bp'],
                        vitals['spo2'],
                        vitals['temperature'],
                        vitals['pulse_rate']
                    ]
                    labels.append(label_vector)
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
        
        return np.array(images), np.array(labels)
    
    def create_model(self):
        """Create a CNN model for vital signs extraction"""
        model = keras.Sequential([
            # Convolutional layers for feature extraction
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
            keras.layers.MaxPooling2D((2, 2)),
            
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            
            keras.layers.Conv2D(256, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            
            # Global average pooling to reduce parameters
            keras.layers.GlobalAveragePooling2D(),
            
            # Dense layers for regression
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.5),
            
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            
            # Output layer for 6 vital signs (regression)
            keras.layers.Dense(6, activation='linear')  # Linear activation for regression
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',  # Mean squared error for regression
            metrics=['mae']  # Mean absolute error
        )
        
        return model
    
    def train_model(self, dataset_dir="generated_heart_monitors", epochs=50):
        """Train the model on the dataset"""
        print("Loading dataset...")
        X, y = self.load_dataset(dataset_dir)
        
        if len(X) == 0:
            raise ValueError("No valid images found in dataset")
        
        print(f"Dataset loaded: {len(X)} images")
        print(f"Image shape: {X[0].shape}")
        print(f"Label shape: {y[0].shape}")
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train)} images")
        print(f"Test set: {len(X_test)} images")
        
        # Create model
        print("Creating model...")
        self.model = self.create_model()
        self.model.summary()
        
        # Train model
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=8,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate model
        print("Evaluating model...")
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        # Save model
        model_path = "TensorFlow/vital_signs_model.h5"
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        return history
    
    def load_model(self, model_path="TensorFlow/vital_signs_model.h5"):
        """Load a trained model"""
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded from: {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    
    def predict_vital_signs(self, image_path):
        """Predict vital signs from a single image"""
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        # Preprocess image
        image = self.preprocess_image(image_path)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Make prediction
        prediction = self.model.predict(image, verbose=0)[0]
        
        # Create result dictionary
        result = {
            "image_path": image_path,
            "timestamp": datetime.now().isoformat(),
            "vital_signs": {}
        }
        
        # Map predictions to labels
        for i, label in enumerate(self.vital_signs_labels):
            if label == 'temperature':
                result["vital_signs"][label] = round(float(prediction[i]), 1)
            else:
                result["vital_signs"][label] = int(round(float(prediction[i])))
        
        # Format blood pressure
        result["vital_signs"]["blood_pressure"] = f"{result['vital_signs']['systolic_bp']}/{result['vital_signs']['diastolic_bp']}"
        
        return result
    
    def predict_batch(self, image_paths, output_file="predictions.json"):
        """Predict vital signs for multiple images and save to JSON"""
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.predict_vital_signs(image_path)
                results.append(result)
                print(f"Processed {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Save results to JSON
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        return results

def main():
    """Main function to train and test the model"""
    extractor = VitalSignsExtractor()
    
    # Check if model exists
    model_path = "vital_signs_model.h5"
    
    if not os.path.exists(model_path):
        print("Training new model...")
        try:
            # Train model
            history = extractor.train_model()
            print("Model training completed!")
        except Exception as e:
            print(f"Error training model: {e}")
            return
    else:
        print("Loading existing model...")
        extractor.load_model(model_path)
    
    # Test on generated images
    dataset_dir = "generated_heart_monitors"
    if os.path.exists(dataset_dir):
        # Get list of image files
        image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.png')]
        image_paths = [os.path.join(dataset_dir, f) for f in image_files[:5]]  # Test on first 5 images
        
        print(f"\nTesting on {len(image_paths)} images...")
        
        # Make predictions
        results = extractor.predict_batch(image_paths, "TensorFlow/vital_signs_predictions.json")
        
        # Display results
        print("\nPrediction Results:")
        for result in results:
            filename = os.path.basename(result['image_path'])
            vitals = result['vital_signs']
            print(f"\n{filename}:")
            print(f"  Heart Rate: {vitals['heart_rate']} bpm")
            print(f"  Blood Pressure: {vitals['blood_pressure']} mmHg")
            print(f"  SpO2: {vitals['spo2']}%")
            print(f"  Temperature: {vitals['temperature']}°C")
            print(f"  Pulse Rate: {vitals['pulse_rate']} /min")
    
    else:
        print(f"Dataset directory not found: {dataset_dir}")
        print("Please run the image generator first to create sample data.")

if __name__ == "__main__":
    main()

