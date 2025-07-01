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
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error

class VitalSignsExtractor:
    def __init__(self):
        self.model = None
        self.image_size = (224, 224)
        self.vital_signs_labels = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'spo2', 'temperature', 'pulse_rate']
        self.best_mae = float('inf')
        self.best_params = None
        self.training_history = []
        
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
    
    def load_dataset(self, dataset_dir=None):
        """Load and prepare the dataset"""
        if dataset_dir is None:
            # Try multiple possible locations for the dataset
            possible_paths = [
                os.path.join("..", "generated_heart_monitors"),
                "generated_heart_monitors",
                os.path.join("..", "..", "generated_heart_monitors")
            ]
            
            dataset_dir = None
            for path in possible_paths:
                metadata_path = os.path.join(path, "dataset_info.json")
                if os.path.exists(metadata_path):
                    dataset_dir = path
                    print(f"Found dataset at: {os.path.abspath(dataset_dir)}")
                    break
            
            if dataset_dir is None:
                raise FileNotFoundError(f"Dataset not found in any of these locations: {possible_paths}")
        
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
    
    def create_model(self, params=None):
        """Create a CNN model for vital signs extraction with configurable parameters"""
        if params is None:
            # Use best known parameters as default
            params = {
                'conv_layers': [32, 64, 128, 256],
                'dense_layers': [512, 256, 128],
                'dropout_rates': [0.248, 0.349],
                'learning_rate': 0.0001,
                'batch_size': 4,
                'activation': 'relu'
            }
        
        model = keras.Sequential()
        
        # Add convolutional layers
        for i, filters in enumerate(params['conv_layers']):
            if i == 0:
                model.add(keras.layers.Conv2D(filters, (3, 3), activation=params['activation'], 
                                            input_shape=(*self.image_size, 3)))
            else:
                model.add(keras.layers.Conv2D(filters, (3, 3), activation=params['activation']))
            model.add(keras.layers.MaxPooling2D((2, 2)))
        
        # Global average pooling
        model.add(keras.layers.GlobalAveragePooling2D())
        
        # Add dense layers with dropout
        for i, units in enumerate(params['dense_layers']):
            model.add(keras.layers.Dense(units, activation=params['activation']))
            if i < len(params['dropout_rates']):
                model.add(keras.layers.Dropout(params['dropout_rates'][i]))
        
        # Output layer for 6 vital signs (regression)
        model.add(keras.layers.Dense(6, activation='linear'))
        
        # Compile model with configurable learning rate
        optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, dataset_dir=None, epochs=50):
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
        
        # Create model with default parameters
        print("Creating model...")
        self.model = self.create_model()  # Uses default parameters
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
        model_path = "vital_signs_model.h5"
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        return history
    
    def load_model(self, model_path="vital_signs_model.h5"):
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

    def generate_random_params(self):
        """Generate random hyperparameters for tuning, focused around best known parameters"""
        # Best known parameters from previous runs
        best_known = {
            "conv_layers": [32, 64, 128, 256],
            "dense_layers": [512, 256, 128],
            "dropout_rates": [0.248, 0.349],
            "learning_rate": 0.0001,
            "batch_size": 4,
            "activation": "relu"
        }
        
        # Generate variations around the best parameters
        conv_options = [
            [32, 64, 128, 256],      # Best known
            [32, 64, 128],           # Simpler version
            [16, 32, 64, 128, 256],  # Deeper version
            [32, 64, 128, 512],      # Different last layer
            [64, 128, 256, 512],     # Larger filters
            [16, 32, 64, 128],       # Medium complexity
        ]
        
        dense_options = [
            [512, 256, 128],         # Best known
            [512, 256],              # Simpler
            [1024, 512, 256],        # Larger
            [256, 128],              # Much simpler
            [512, 256, 128, 64],     # Deeper
            [768, 384, 192],         # Alternative sizes
        ]
        
        # Learning rates around the best known value
        learning_rates = [0.0001, 0.00005, 0.0002, 0.00015, 0.00008]
        
        # Batch sizes around the best known value
        batch_sizes = [4, 8, 2, 6]  # Focus on smaller batch sizes
        
        # Dropout rates around the best known values
        dropout_rate_1 = random.uniform(0.2, 0.35)  # Around 0.248
        dropout_rate_2 = random.uniform(0.25, 0.45)  # Around 0.349
        
        params = {
            'conv_layers': random.choice(conv_options),
            'dense_layers': random.choice(dense_options),
            'dropout_rates': [dropout_rate_1, dropout_rate_2],
            'learning_rate': random.choice(learning_rates),
            'batch_size': random.choice(batch_sizes),
            'activation': random.choice(['relu', 'elu'])  # Focus on better activations
        }
        
        return params
    
    def load_best_params_if_available(self):
        """Load best parameters from previous runs if available"""
        try:
            if os.path.exists("best_hyperparameters.json"):
                with open("best_hyperparameters.json", 'r') as f:
                    best_data = json.load(f)
                    self.best_params = best_data['params']
                    self.best_mae = best_data['mae']
                    print(f"📋 Loaded previous best parameters (MAE: {self.best_mae:.4f})")
                    print(f"   Parameters: {self.best_params}")
                    return True
        except Exception as e:
            print(f"Could not load previous best parameters: {e}")
        return False
    
    def get_refined_params_around_best(self):
        """Generate parameters that are small variations of the current best"""
        if self.best_params is None:
            return self.generate_random_params()
        
        # Make small adjustments to the best parameters
        base_params = self.best_params.copy()
        
        # Small variations
        variations = {
            'learning_rate': [
                base_params['learning_rate'] * 0.5,
                base_params['learning_rate'] * 0.8,
                base_params['learning_rate'],
                base_params['learning_rate'] * 1.2,
                base_params['learning_rate'] * 1.5
            ],
            'batch_size': [2, 4, 6, 8],
            'dropout_adjustments': [
                [base_params['dropout_rates'][0] * 0.8, base_params['dropout_rates'][1] * 0.8],
                [base_params['dropout_rates'][0] * 0.9, base_params['dropout_rates'][1] * 0.9],
                base_params['dropout_rates'],
                [base_params['dropout_rates'][0] * 1.1, base_params['dropout_rates'][1] * 1.1],
                [base_params['dropout_rates'][0] * 1.2, base_params['dropout_rates'][1] * 1.2]
            ]
        }
        
        refined_params = base_params.copy()
        refined_params['learning_rate'] = random.choice(variations['learning_rate'])
        refined_params['batch_size'] = random.choice(variations['batch_size'])
        refined_params['dropout_rates'] = random.choice(variations['dropout_adjustments'])
        
        # Ensure dropout rates are within valid range
        refined_params['dropout_rates'] = [
            max(0.1, min(0.7, refined_params['dropout_rates'][0])),
            max(0.1, min(0.7, refined_params['dropout_rates'][1]))
        ]
        
        return refined_params
    
    def evaluate_predictions_accuracy(self, dataset_dir=None):
        """Evaluate how close predictions are to ground truth"""
        if dataset_dir is None:
            # Try multiple possible locations for the dataset
            possible_paths = [
                os.path.join("..", "generated_heart_monitors"),
                "generated_heart_monitors",
                os.path.join("..", "..", "generated_heart_monitors")
            ]
            
            dataset_dir = None
            for path in possible_paths:
                metadata_path = os.path.join(path, "dataset_info.json")
                if os.path.exists(metadata_path):
                    dataset_dir = path
                    break
            
            if dataset_dir is None:
                raise FileNotFoundError(f"Dataset not found in any of these locations: {possible_paths}")
        
        # Load ground truth data
        metadata_path = os.path.join(dataset_dir, "dataset_info.json")
        with open(metadata_path, 'r') as f:
            dataset_info = json.load(f)
        
        # Get image files for prediction
        image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.png')]
        image_paths = [os.path.join(dataset_dir, f) for f in image_files]
        
        # Make predictions
        predictions = []
        ground_truth = []
        
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            
            # Find corresponding ground truth
            gt_item = next((item for item in dataset_info if item['filename'] == filename), None)
            if gt_item is None:
                continue
                
            try:
                # Get prediction
                result = self.predict_vital_signs(image_path)
                pred_vitals = result['vital_signs']
                
                # Extract values for comparison
                pred_values = [
                    pred_vitals['heart_rate'],
                    pred_vitals['systolic_bp'],
                    pred_vitals['diastolic_bp'],
                    pred_vitals['spo2'],
                    pred_vitals['temperature'],
                    pred_vitals['pulse_rate']
                ]
                
                gt_values = [
                    gt_item['vitals']['heart_rate'],
                    gt_item['vitals']['systolic_bp'],
                    gt_item['vitals']['diastolic_bp'],
                    gt_item['vitals']['spo2'],
                    gt_item['vitals']['temperature'],
                    gt_item['vitals']['pulse_rate']
                ]
                
                predictions.append(pred_values)
                ground_truth.append(gt_values)
                
            except Exception as e:
                print(f"Error evaluating {filename}: {e}")
                continue
        
        if not predictions:
            return float('inf'), {}
        
        # Calculate metrics
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        mae = mean_absolute_error(ground_truth, predictions)
        mse = mean_squared_error(ground_truth, predictions)
        
        # Calculate per-vital-sign accuracy
        vital_accuracy = {}
        for i, label in enumerate(self.vital_signs_labels):
            vital_mae = mean_absolute_error(ground_truth[:, i], predictions[:, i])
            vital_accuracy[label] = vital_mae
        
        return mae, {'mae': mae, 'mse': mse, 'vital_accuracy': vital_accuracy}
    
    def train_with_hyperparameter_tuning(self, dataset_dir=None, 
                                       max_iterations=20, target_mae=5.0, epochs=30):
        """Train model with automatic hyperparameter tuning"""
        print(f"Starting hyperparameter tuning with {max_iterations} iterations...")
        print(f"Target MAE: {target_mae}")
        
        # Load dataset once
        X, y = self.load_dataset(dataset_dir)
        if len(X) == 0:
            raise ValueError("No valid images found in dataset")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Load previous best parameters if available
        self.load_best_params_if_available()
        
        for iteration in range(max_iterations):
            print(f"\n{'='*50}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*50}")
            
            # Use different strategies for parameter generation
            if iteration < 5 and self.best_params is not None:
                # First 5 iterations: refine around best known parameters
                params = self.get_refined_params_around_best()
                print("🎯 Using refined parameters around best known values")
            elif iteration < 10:
                # Next 5 iterations: focused random search
                params = self.generate_random_params()
                print("🔍 Using focused random search")
            else:
                # Remaining iterations: broader exploration
                params = self.generate_random_params()
                print("🌐 Using broader parameter exploration")
            
            print(f"Testing parameters: {params}")
            
            try:
                # Create and train model
                self.model = self.create_model(params)
                
                # Train with early stopping and learning rate reduction
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor='val_mae', patience=8, restore_best_weights=True
                )
                
                # Reduce learning rate on plateau
                reduce_lr = keras.callbacks.ReduceLROnPlateau(
                    monitor='val_mae', factor=0.5, patience=5, min_lr=1e-7, verbose=0
                )
                
                history = self.model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=params['batch_size'],
                    validation_data=(X_test, y_test),
                    verbose=0,  # Reduce output
                    callbacks=[early_stopping, reduce_lr]
                )
                
                # Evaluate on test set
                test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
                
                # Evaluate prediction accuracy against ground truth
                prediction_mae, detailed_metrics = self.evaluate_predictions_accuracy(dataset_dir)
                
                print(f"Test MAE: {test_mae:.4f}")
                print(f"Prediction MAE vs Ground Truth: {prediction_mae:.4f}")
                
                # Track this iteration
                iteration_result = {
                    'iteration': iteration + 1,
                    'params': params,
                    'test_mae': test_mae,
                    'prediction_mae': prediction_mae,
                    'detailed_metrics': detailed_metrics
                }
                self.training_history.append(iteration_result)
                
                # Check if this is the best model so far
                if prediction_mae < self.best_mae:
                    self.best_mae = prediction_mae
                    self.best_params = params.copy()
                    
                    # Save best model
                    model_path = "vital_signs_model_best.h5"
                    self.model.save(model_path)
                    print(f"🎉 NEW BEST MODEL! MAE: {prediction_mae:.4f}")
                    print(f"Model saved to: {model_path}")
                    
                    # Save best parameters
                    with open("best_hyperparameters.json", 'w') as f:
                        json.dump({
                            'params': self.best_params,
                            'mae': self.best_mae,
                            'iteration': iteration + 1,
                            'detailed_metrics': detailed_metrics
                        }, f, indent=2)
                    
                    # Check if target achieved
                    if prediction_mae <= target_mae:
                        print(f"🎯 TARGET MAE ACHIEVED! ({prediction_mae:.4f} <= {target_mae})")
                        break
                else:
                    print(f"Current best MAE: {self.best_mae:.4f}")
                
                # Display per-vital accuracy for best model
                if prediction_mae == self.best_mae:
                    print("\nPer-vital sign accuracy (MAE):")
                    for vital, accuracy in detailed_metrics['vital_accuracy'].items():
                        print(f"  {vital}: {accuracy:.2f}")
                
            except Exception as e:
                print(f"Error in iteration {iteration + 1}: {e}")
                continue
        
        print(f"\n{'='*50}")
        print("HYPERPARAMETER TUNING COMPLETE")
        print(f"{'='*50}")
        print(f"Best MAE achieved: {self.best_mae:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        # Load the best model
        if os.path.exists("vital_signs_model_best.h5"):
            self.model = keras.models.load_model("vital_signs_model_best.h5")
            print("Best model loaded for final predictions.")
        
        # Save training history
        with open("training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        return self.best_params, self.best_mae

def find_dataset_directory():
    """Helper function to find the dataset directory"""
    possible_paths = [
        os.path.join("..", "generated_heart_monitors"),
        "generated_heart_monitors", 
        os.path.join("..", "..", "generated_heart_monitors"),
        os.path.join(os.getcwd(), "..", "generated_heart_monitors"),
        r"C:\Users\junha\OneDrive\Documents\GitHub\foundation-internship\generated_heart_monitors"
    ]
    
    print("🔍 Searching for dataset in these locations:")
    for i, path in enumerate(possible_paths):
        abs_path = os.path.abspath(path)
        metadata_path = os.path.join(path, "dataset_info.json")
        exists = os.path.exists(metadata_path)
        print(f"  {i+1}. {abs_path} - {'✅ FOUND' if exists else '❌ Not found'}")
        
        if exists:
            return path
    
    return None

def main():
    """Main function to train and test the model with hyperparameter tuning"""
    extractor = VitalSignsExtractor()
    
    # Check if best model exists
    model_path = "vital_signs_model_best.h5"
    
    # Ask user if they want to run hyperparameter tuning
    print("=== VITAL SIGNS EXTRACTOR ===")
    print("Choose an option:")
    print("1. Run hyperparameter tuning (automatic parameter optimization)")
    print("2. Load existing best model and make predictions")
    print("3. Train with default parameters")
    
    try:
        choice = input("Enter your choice (1/2/3): ").strip()
    except:
        choice = "1"  # Default to hyperparameter tuning
    
    if choice == "1":
        print("\n🚀 Starting hyperparameter tuning...")
        try:
            # Run hyperparameter tuning with improved settings
            best_params, best_mae = extractor.train_with_hyperparameter_tuning(
                dataset_dir=None,  # Auto-detect dataset location
                max_iterations=30,  # More iterations to find better parameters
                target_mae=2.0,     # More aggressive target (lower MAE)
                epochs=75           # More epochs per iteration
            )
            print(f"\n✅ Hyperparameter tuning completed!")
            print(f"Best MAE: {best_mae:.4f}")
            
        except Exception as e:
            print(f"Error during hyperparameter tuning: {e}")
            return
            
    elif choice == "2":
        if os.path.exists(model_path):
            print("Loading existing best model...")
            extractor.model = keras.models.load_model(model_path)
            print("✅ Best model loaded!")
        else:
            print("❌ No best model found. Running hyperparameter tuning instead...")
            try:
                best_params, best_mae = extractor.train_with_hyperparameter_tuning(
                    dataset_dir="../generated_heart_monitors"
                )
            except Exception as e:
                print(f"Error: {e}")
                return
    
    else:  # choice == "3"
        print("Training with default parameters...")
        try:
            history = extractor.train_model(
                dataset_dir="../generated_heart_monitors",
                epochs=50
            )
        except Exception as e:
            print(f"Error training model: {e}")
            return
    
    # Test on generated images - auto-detect dataset location
    possible_dataset_paths = [
        os.path.join("..", "generated_heart_monitors"),
        "generated_heart_monitors",
        os.path.join("..", "..", "generated_heart_monitors")
    ]
    
    dataset_dir = None
    for path in possible_dataset_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "dataset_info.json")):
            dataset_dir = path
            print(f"Found dataset at: {os.path.abspath(dataset_dir)}")
            break
    
    if dataset_dir:
        # Get list of image files
        image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.png')]
        image_paths = [os.path.join(dataset_dir, f) for f in image_files[:10]]  # Test on first 10 images
        
        print(f"\n📊 Testing on {len(image_paths)} images...")
        
        # Make predictions
        results = extractor.predict_batch(image_paths, "vital_signs_predictions.json")
        
        # Load ground truth for comparison
        metadata_path = os.path.join(dataset_dir, "dataset_info.json")
        with open(metadata_path, 'r') as f:
            ground_truth = json.load(f)
        
        # Create comparison report
        comparison_report = []
        
        print("\n📈 PREDICTION vs GROUND TRUTH COMPARISON:")
        print("="*80)
        
        for result in results:
            filename = os.path.basename(result['image_path'])
            pred_vitals = result['vital_signs']
            
            # Find ground truth
            gt_item = next((item for item in ground_truth if item['filename'] == filename), None)
            if gt_item:
                gt_vitals = gt_item['vitals']
                
                print(f"\n📄 {filename}:")
                print(f"  Heart Rate:    Pred: {pred_vitals['heart_rate']:3d} | GT: {gt_vitals['heart_rate']:3d} | Diff: {abs(pred_vitals['heart_rate'] - gt_vitals['heart_rate']):3d}")
                print(f"  Systolic BP:   Pred: {pred_vitals['systolic_bp']:3d} | GT: {gt_vitals['systolic_bp']:3d} | Diff: {abs(pred_vitals['systolic_bp'] - gt_vitals['systolic_bp']):3d}")
                print(f"  Diastolic BP:  Pred: {pred_vitals['diastolic_bp']:3d} | GT: {gt_vitals['diastolic_bp']:3d} | Diff: {abs(pred_vitals['diastolic_bp'] - gt_vitals['diastolic_bp']):3d}")
                print(f"  SpO2:          Pred: {pred_vitals['spo2']:3d} | GT: {gt_vitals['spo2']:3d} | Diff: {abs(pred_vitals['spo2'] - gt_vitals['spo2']):3d}")
                print(f"  Temperature:   Pred: {pred_vitals['temperature']:5.1f} | GT: {gt_vitals['temperature']:5.1f} | Diff: {abs(pred_vitals['temperature'] - gt_vitals['temperature']):5.1f}")
                print(f"  Pulse Rate:    Pred: {pred_vitals['pulse_rate']:3d} | GT: {gt_vitals['pulse_rate']:3d} | Diff: {abs(pred_vitals['pulse_rate'] - gt_vitals['pulse_rate']):3d}")
                
                # Add to comparison report
                comparison_report.append({
                    'filename': filename,
                    'predictions': pred_vitals,
                    'ground_truth': gt_vitals,
                    'absolute_errors': {
                        'heart_rate': abs(pred_vitals['heart_rate'] - gt_vitals['heart_rate']),
                        'systolic_bp': abs(pred_vitals['systolic_bp'] - gt_vitals['systolic_bp']),
                        'diastolic_bp': abs(pred_vitals['diastolic_bp'] - gt_vitals['diastolic_bp']),
                        'spo2': abs(pred_vitals['spo2'] - gt_vitals['spo2']),
                        'temperature': abs(pred_vitals['temperature'] - gt_vitals['temperature']),
                        'pulse_rate': abs(pred_vitals['pulse_rate'] - gt_vitals['pulse_rate'])
                    }
                })
        
        # Save comparison report
        with open("prediction_comparison.json", 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        # Calculate overall accuracy
        if comparison_report:
            total_errors = {vital: 0 for vital in extractor.vital_signs_labels}
            for report in comparison_report:
                for vital in extractor.vital_signs_labels:
                    total_errors[vital] += report['absolute_errors'][vital]
            
            avg_errors = {vital: total_errors[vital] / len(comparison_report) 
                         for vital in extractor.vital_signs_labels}
            
            print(f"\n📊 AVERAGE ABSOLUTE ERRORS:")
            print("="*40)
            for vital, error in avg_errors.items():
                print(f"  {vital}: {error:.2f}")
            
            overall_mae = sum(avg_errors.values()) / len(avg_errors)
            print(f"\n🎯 OVERALL MAE: {overall_mae:.2f}")
        
        print(f"\n💾 Files saved:")
        print(f"  - vital_signs_predictions.json")
        print(f"  - prediction_comparison.json")
        if extractor.best_params:
            print(f"  - best_hyperparameters.json")
            print(f"  - training_history.json")
            print(f"  - vital_signs_model_best.h5")
    
    else:
        print(f"❌ Dataset directory not found: {dataset_dir}")
        print("Please ensure the generated heart monitor images are in the correct location.")

if __name__ == "__main__":
    main()

