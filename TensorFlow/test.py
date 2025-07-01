# Medical Monitor Image → CNN → 6 Numerical Values

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

# TensorFlow Performance Optimization
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Configure TensorFlow for optimal CPU performance
tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores

# Enable mixed precision for better performance (if GPU available)
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("✅ Mixed precision enabled for better performance")
except:
    print("ℹ️ Mixed precision not available, using default precision")

# Enable memory growth for GPU (if available)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Enable CPU optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations

# Configure for optimal CPU performance
if not gpus:
    # CPU-specific optimizations
    tf.config.optimizer.set_jit(True)  # Enable XLA compilation
    print("✅ CPU optimizations enabled (XLA, oneDNN)")

class VitalSignsExtractor:
    def __init__(self):
        self.model = None
        self.image_size = (224, 224)
        self.vital_signs_labels = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'spo2', 'temperature', 'pulse_rate']
        self.best_mae = float('inf')
        self.best_params = None
        self.training_history = []
        
        # Performance optimization settings
        self.use_mixed_precision = tf.keras.mixed_precision.global_policy().name == 'mixed_float16'
        self.parallel_calls = tf.data.AUTOTUNE
        
    def preprocess_image(self, image_path):
        """Preprocess image for the model with optimized performance"""
        # Read and resize image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size using optimized interpolation
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def create_optimized_dataset(self, X, y, batch_size, is_training=True):
        """Create optimized tf.data.Dataset for better performance"""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if is_training:
            # Shuffle for training
            dataset = dataset.shuffle(buffer_size=min(1000, len(X)))
        
        # Batch the dataset
        dataset = dataset.batch(batch_size)
        
        # Prefetch for performance
        dataset = dataset.prefetch(self.parallel_calls)
        
        return dataset
    
    def load_dataset(self, dataset_dir=None):
        """Load and prepare the dataset with parallel processing"""
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
        
        print(f"Loading {len(dataset_info)} images with parallel processing...")
        
        # Use multiprocessing for faster image loading
        from concurrent.futures import ThreadPoolExecutor
        import functools
        
        def process_item(item):
            image_path = os.path.join(dataset_dir, item['filename'])
            if os.path.exists(image_path):
                try:
                    # Load and preprocess image
                    image = self.preprocess_image(image_path)
                    
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
                    return image, label_vector
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    return None
            return None
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(process_item, dataset_info))
        
        # Filter out None results and separate images and labels
        for result in results:
            if result is not None:
                image, label = result
                images.append(image)
                labels.append(label)
        
        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)
    
    def create_model(self, params=None):
        """Create a CNN model with performance optimizations"""
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
        
        # Use functional API for better performance
        inputs = keras.layers.Input(shape=(*self.image_size, 3))
        x = inputs
        
        # Add convolutional layers with batch normalization
        for i, filters in enumerate(params['conv_layers']):
            x = keras.layers.Conv2D(filters, (3, 3), activation=params['activation'], 
                                  padding='same', use_bias=False)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.MaxPooling2D((2, 2))(x)
        
        # Global average pooling (more efficient than flatten)
        x = keras.layers.GlobalAveragePooling2D()(x)
        
        # Add dense layers with dropout
        for i, units in enumerate(params['dense_layers']):
            x = keras.layers.Dense(units, activation=params['activation'])(x)
            if i < len(params['dropout_rates']):
                x = keras.layers.Dropout(params['dropout_rates'][i])(x)
        
        # Output layer for 6 vital signs (regression)
        if self.use_mixed_precision:
            outputs = keras.layers.Dense(6, activation='linear', dtype='float32')(x)
        else:
            outputs = keras.layers.Dense(6, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model with optimized settings and proper loss function reference
        optimizer = keras.optimizers.Adam(
            learning_rate=params['learning_rate'],
            epsilon=1e-7,  # Prevent numerical instability with mixed precision
            clipnorm=1.0   # Gradient clipping for stability
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',  # Use full name instead of 'mse' to avoid serialization issues
            metrics=['mean_absolute_error']  # Use full name instead of 'mae'
        )
        
        return model
    
    def train_model(self, dataset_dir=None, epochs=50):
        """Train the model with performance optimizations"""
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
        
        # Create optimized datasets
        train_dataset = self.create_optimized_dataset(X_train, y_train, 8, is_training=True)
        test_dataset = self.create_optimized_dataset(X_test, y_test, 8, is_training=False)
        
        # Create model with default parameters
        print("Creating optimized model...")
        self.model = self.create_model()
        self.model.summary()
        
        # Enhanced callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_mae', patience=10, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_mae', factor=0.5, patience=5, min_lr=1e-7, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'vital_signs_model_checkpoint.h5', 
                monitor='val_mae', save_best_only=True, verbose=1
            )
        ]
        
        # Train model with optimized dataset
        print("Training model with performance optimizations...")
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=test_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        print("Evaluating model...")
        test_loss, test_mae = self.model.evaluate(test_dataset, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        # Save model
        model_path = "vital_signs_model.h5"
        self.save_model_safely(model_path)
        
        return history
    
    def load_model(self, model_path="vital_signs_model.h5"):
        """Load a trained model with error handling"""
        if os.path.exists(model_path):
            try:
                # Try loading with compile=True first
                self.model = keras.models.load_model(model_path)
                print(f"✅ Model loaded from: {model_path}")
                
                # Validate model by checking if it can make a prediction
                test_input = np.random.random((1, *self.image_size, 3)).astype(np.float32)
                test_output = self.model.predict(test_input, verbose=0)
                if test_output.shape != (1, 6):
                    raise ValueError("Model output shape is incorrect")
                    
            except Exception as e:
                print(f"⚠️ Error loading compiled model: {e}")
                print("🔄 Attempting to load model without compilation...")
                
                try:
                    # Try loading without compilation and recompile
                    self.model = keras.models.load_model(model_path, compile=False)
                    
                    # Recompile the model with proper settings
                    optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-7, clipnorm=1.0)
                    self.model.compile(
                        optimizer=optimizer,
                        loss='mean_squared_error',
                        metrics=['mean_absolute_error']
                    )
                    
                    print(f"✅ Model loaded and recompiled from: {model_path}")
                    
                except Exception as e2:
                    print(f"❌ Failed to load model: {e2}")
                    raise FileNotFoundError(f"Could not load model from: {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    
    def save_model_safely(self, model_path, model=None):
        """Save model with error handling and backup options"""
        if model is None:
            model = self.model
            
        if model is None:
            raise ValueError("No model to save")
        
        try:
            # Primary save method
            model.save(model_path)
            print(f"✅ Model saved to: {model_path}")
            
            # Also save weights separately as backup
            weights_path = model_path.replace('.h5', '_weights.h5')
            model.save_weights(weights_path)
            print(f"✅ Model weights saved to: {weights_path}")
            
            # Save model architecture as backup
            architecture_path = model_path.replace('.h5', '_architecture.json')
            with open(architecture_path, 'w') as f:
                f.write(model.to_json())
            print(f"✅ Model architecture saved to: {architecture_path}")
            
        except Exception as e:
            print(f"⚠️ Error saving model: {e}")
            # Fallback: save only weights and architecture
            try:
                weights_path = model_path.replace('.h5', '_weights_backup.h5')
                model.save_weights(weights_path)
                
                architecture_path = model_path.replace('.h5', '_architecture_backup.json')
                with open(architecture_path, 'w') as f:
                    f.write(model.to_json())
                    
                print(f"⚠️ Model saved with fallback method (weights + architecture)")
                return False
            except Exception as e2:
                print(f"❌ Failed to save model: {e2}")
                raise
        
        return True
    
    def load_model_from_backup(self, model_path):
        """Load model from backup files (weights + architecture)"""
        architecture_path = model_path.replace('.h5', '_architecture.json')
        weights_path = model_path.replace('.h5', '_weights.h5')
        
        # Try backup files if main files don't exist
        if not os.path.exists(architecture_path):
            architecture_path = model_path.replace('.h5', '_architecture_backup.json')
        if not os.path.exists(weights_path):
            weights_path = model_path.replace('.h5', '_weights_backup.h5')
        
        if os.path.exists(architecture_path) and os.path.exists(weights_path):
            try:
                # Load architecture
                with open(architecture_path, 'r') as f:
                    model_json = f.read()
                
                # Create model from architecture
                self.model = keras.models.model_from_json(model_json)
                
                # Load weights
                self.model.load_weights(weights_path)
                
                # Recompile model
                optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-7, clipnorm=1.0)
                self.model.compile(
                    optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['mean_absolute_error']
                )
                
                print(f"✅ Model loaded from backup files (architecture + weights)")
                return True
                
            except Exception as e:
                print(f"❌ Failed to load from backup: {e}")
                return False
        
        return False

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
        """Train model with automatic hyperparameter tuning and performance optimizations"""
        print(f"Starting optimized hyperparameter tuning with {max_iterations} iterations...")
        print(f"Target MAE: {target_mae}")
        
        # Load dataset once with parallel processing
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
                params = self.get_refined_params_around_best()
                print("🎯 Using refined parameters around best known values")
            elif iteration < 10:
                params = self.generate_random_params()
                print("🔍 Using focused random search")
            else:
                params = self.generate_random_params()
                print("🌐 Using broader parameter exploration")
            
            print(f"Testing parameters: {params}")
            
            try:
                # Create optimized datasets for this iteration
                train_dataset = self.create_optimized_dataset(
                    X_train, y_train, params['batch_size'], is_training=True
                )
                test_dataset = self.create_optimized_dataset(
                    X_test, y_test, params['batch_size'], is_training=False
                )
                
                # Create and train model
                self.model = self.create_model(params)
                
                # Enhanced callbacks
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor='val_mae', patience=8, restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_mae', factor=0.5, patience=5, min_lr=1e-7
                    )
                ]
                
                history = self.model.fit(
                    train_dataset,
                    epochs=epochs,
                    validation_data=test_dataset,
                    verbose=0,
                    callbacks=callbacks
                )
                
                # Evaluate on test set
                test_loss, test_mae = self.model.evaluate(test_dataset, verbose=0)
                
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
                    'detailed_metrics': detailed_metrics,
                    'performance_optimized': True
                }
                self.training_history.append(iteration_result)
                
                # Check if this is the best model so far
                if prediction_mae < self.best_mae:
                    self.best_mae = prediction_mae
                    self.best_params = params.copy()
                    
                    # Save best model safely
                    model_path = "vital_signs_model_best.h5"
                    self.save_model_safely(model_path)
                    print(f"🎉 NEW BEST MODEL! MAE: {prediction_mae:.4f}")
                    
                    # Save best parameters
                    with open("best_hyperparameters.json", 'w') as f:
                        json.dump({
                            'params': self.best_params,
                            'mae': self.best_mae,
                            'iteration': iteration + 1,
                            'detailed_metrics': detailed_metrics,
                            'performance_optimized': True
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
        
        # Load the best model safely
        if os.path.exists("vital_signs_model_best.h5"):
            try:
                self.load_model("vital_signs_model_best.h5")
                print("✅ Best model loaded for final predictions.")
            except:
                if self.load_model_from_backup("vital_signs_model_best.h5"):
                    print("✅ Best model loaded from backup files.")
                else:
                    print("⚠️ Could not load best model, using current model.")
        
        # Save training history
        with open("training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        return self.best_params, self.best_mae

    def comprehensive_model_test(self, dataset_dir=None, test_subset_size=None):
        """Comprehensive test of model performance against ground truth data"""
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        print("🔬 COMPREHENSIVE MODEL TESTING")
        print("="*60)
        
        # Auto-detect dataset if not provided
        if dataset_dir is None:
            possible_paths = [
                os.path.join("..", "generated_heart_monitors"),
                "generated_heart_monitors",
                os.path.join("..", "..", "generated_heart_monitors")
            ]
            
            for path in possible_paths:
                metadata_path = os.path.join(path, "dataset_info.json")
                if os.path.exists(metadata_path):
                    dataset_dir = path
                    break
            
            if dataset_dir is None:
                raise FileNotFoundError("Dataset not found in expected locations")
        
        # Load ground truth data
        metadata_path = os.path.join(dataset_dir, "dataset_info.json")
        with open(metadata_path, 'r') as f:
            dataset_info = json.load(f)
        
        print(f"📊 Found {len(dataset_info)} samples in dataset")
        print(f"📁 Dataset location: {os.path.abspath(dataset_dir)}")
        
        # Limit test size if specified
        if test_subset_size and test_subset_size < len(dataset_info):
            dataset_info = random.sample(dataset_info, test_subset_size)
            print(f"🎯 Testing on random subset of {len(dataset_info)} samples")
        
        # Initialize tracking variables
        predictions = []
        ground_truth = []
        detailed_results = []
        scenario_errors = {}
        processing_errors = 0
        
        print("\n🔄 Processing images and making predictions...")
        
        # Process each image
        for i, item in enumerate(dataset_info):
            image_path = os.path.join(dataset_dir, item['filename'])
            
            if not os.path.exists(image_path):
                print(f"⚠️ Image not found: {item['filename']}")
                processing_errors += 1
                continue
            
            try:
                # Make prediction
                result = self.predict_vital_signs(image_path)
                pred_vitals = result['vital_signs']
                gt_vitals = item['vitals']

                # Extract prediction values
                pred_values = [
                    pred_vitals['heart_rate'],
                    pred_vitals['systolic_bp'],
                    pred_vitals['diastolic_bp'],
                    pred_vitals['spo2'],
                    pred_vitals['temperature'],
                    pred_vitals['pulse_rate']
                ]
                
                # Extract ground truth values
                gt_values = [
                    gt_vitals['heart_rate'],
                    gt_vitals['systolic_bp'],
                    gt_vitals['diastolic_bp'],
                    gt_vitals['spo2'],
                    gt_vitals['temperature'],
                    gt_vitals['pulse_rate']
                ]
                
                predictions.append(pred_values)
                ground_truth.append(gt_values)
                
                # Calculate individual errors
                errors = [abs(p - g) for p, g in zip(pred_values, gt_values)]
                
                # Store detailed results
                detailed_result = {
                    'filename': item['filename'],
                    'scenario': item['scenario'],
                    'patient_id': item['patient_id'],
                    'predictions': dict(zip(self.vital_signs_labels, pred_values)),
                    'ground_truth': dict(zip(self.vital_signs_labels, gt_values)),
                    'absolute_errors': dict(zip(self.vital_signs_labels, errors)),
                    'overall_mae': sum(errors) / len(errors)
                }
                detailed_results.append(detailed_result)
                
                # Track errors by scenario
                scenario = item['scenario']
                if scenario not in scenario_errors:
                    scenario_errors[scenario] = []
                scenario_errors[scenario].append(errors)
                
                # Progress indicator
                if (i + 1) % 5 == 0 or (i + 1) == len(dataset_info):
                    print(f"  Processed {i + 1}/{len(dataset_info)} images...")
                
            except Exception as e:
                print(f"⚠️ Error processing {item['filename']}: {e}")
                processing_errors += 1
                continue
        
        if not predictions:
            print("❌ No valid predictions were made")
            return None
        
        # Convert to numpy arrays for analysis
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        print(f"\n✅ Successfully processed {len(predictions)} images")
        if processing_errors > 0:
            print(f"⚠️ Failed to process {processing_errors} images")
        
        # Calculate overall metrics
        overall_mae = mean_absolute_error(ground_truth, predictions)
        overall_mse = mean_squared_error(ground_truth, predictions)
        overall_rmse = np.sqrt(overall_mse)
        
        print(f"\n📈 OVERALL PERFORMANCE METRICS")
        print("="*40)
        print(f"  Mean Absolute Error (MAE): {overall_mae:.3f}")
        print(f"  Mean Squared Error (MSE):  {overall_mse:.3f}")
        print(f"  Root Mean Squared Error:   {overall_rmse:.3f}")
        
        # Per-vital sign analysis
        print(f"\n📊 PER-VITAL SIGN PERFORMANCE")
        print("="*50)
        vital_metrics = {}
        
        for i, vital in enumerate(self.vital_signs_labels):
            vital_mae = mean_absolute_error(ground_truth[:, i], predictions[:, i])
            vital_mse = mean_squared_error(ground_truth[:, i], predictions[:, i])
            vital_rmse = np.sqrt(vital_mse)
            
            # Calculate percentage error for non-temperature vitals
            if vital != 'temperature':
                mean_gt_value = np.mean(ground_truth[:, i])
                percentage_error = (vital_mae / mean_gt_value) * 100
            else:
                # For temperature, calculate based on normal range
                percentage_error = (vital_mae / 37.0) * 100
            
            vital_metrics[vital] = {
                'mae': vital_mae,
                'mse': vital_mse,
                'rmse': vital_rmse,
                'percentage_error': percentage_error,
                'mean_ground_truth': np.mean(ground_truth[:, i]),
                'mean_prediction': np.mean(predictions[:, i])
            }
            
            print(f"  {vital.upper().replace('_', ' '):<15}: MAE={vital_mae:6.2f} | RMSE={vital_rmse:6.2f} | Error={percentage_error:5.1f}%")
        
        # Scenario-based analysis
        print(f"\n🎭 PERFORMANCE BY SCENARIO")
        print("="*40)
        scenario_metrics = {}
        
        for scenario, error_lists in scenario_errors.items():
            scenario_errors_array = np.array(error_lists)
            scenario_mae = np.mean(scenario_errors_array)
            scenario_count = len(error_lists)
            
            scenario_metrics[scenario] = {
                'mae': scenario_mae,
                'count': scenario_count,
                'per_vital_mae': np.mean(scenario_errors_array, axis=0)
            }
            
            print(f"  {scenario.upper():<15}: MAE={scenario_mae:6.2f} | Samples={scenario_count:3d}")
        
        # Best and worst predictions
        print(f"\n🏆 BEST AND WORST PREDICTIONS")
        print("="*45)
        
        # Sort by overall MAE
        detailed_results.sort(key=lambda x: x['overall_mae'])
        
        print("🥇 BEST PREDICTIONS (Lowest MAE):")
        for i, result in enumerate(detailed_results[:3]):
            print(f"  {i+1}. {result['filename']} ({result['scenario']})")
            print(f"     MAE: {result['overall_mae']:.2f}")
            print(f"     HR: {result['predictions']['heart_rate']} vs {result['ground_truth']['heart_rate']} (diff: {result['absolute_errors']['heart_rate']})")
        
        print("\n🥉 WORST PREDICTIONS (Highest MAE):")
        for i, result in enumerate(detailed_results[-3:]):
            print(f"  {i+1}. {result['filename']} ({result['scenario']})")
            print(f"     MAE: {result['overall_mae']:.2f}")
            print(f"     HR: {result['predictions']['heart_rate']} vs {result['ground_truth']['heart_rate']} (diff: {result['absolute_errors']['heart_rate']})")
        
        # Error distribution analysis
        print(f"\n📈 ERROR DISTRIBUTION ANALYSIS")
        print("="*45)
        
        overall_errors = [result['overall_mae'] for result in detailed_results]
        print(f"  Error Range: {min(overall_errors):.2f} - {max(overall_errors):.2f}")
        print(f"  Error Std Dev: {np.std(overall_errors):.2f}")
        print(f"  Median Error: {np.median(overall_errors):.2f}")
        
        # Error percentiles
        percentiles = [25, 50, 75, 90, 95]
        print("  Error Percentiles:")
        for p in percentiles:
            value = np.percentile(overall_errors, p)
            print(f"    {p}th percentile: {value:.2f}")
        
        # Save detailed test results
        test_report = {
            'test_summary': {
                'total_samples': len(dataset_info),
                'processed_samples': len(predictions),
                'processing_errors': processing_errors,
                'overall_mae': overall_mae,
                'overall_mse': overall_mse,
                'overall_rmse': overall_rmse
            },
            'vital_metrics': vital_metrics,
            'scenario_metrics': scenario_metrics,
            'detailed_results': detailed_results,
            'error_statistics': {
                'min_error': float(min(overall_errors)),
                'max_error': float(max(overall_errors)),
                'std_error': float(np.std(overall_errors)),
                'median_error': float(np.median(overall_errors)),
                'percentiles': {f'{p}th': float(np.percentile(overall_errors, p)) for p in percentiles}
            },
            'test_timestamp': datetime.now().isoformat()
        }
        
        # Save test report
        report_filename = f"model_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(test_report, f, indent=2)
        
        print(f"\n💾 Detailed test report saved to: {report_filename}")
        
        # Performance assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT")
        print("="*35)
        
        # Define performance thresholds
        excellent_threshold = 2.0
        good_threshold = 5.0
        acceptable_threshold = 10.0
        
        if overall_mae <= excellent_threshold:
            assessment = "🌟 EXCELLENT"
            color_emoji = "🟢"
        elif overall_mae <= good_threshold:
            assessment = "👍 GOOD"
            color_emoji = "🟡"
        elif overall_mae <= acceptable_threshold:
            assessment = "⚠️ ACCEPTABLE"
            color_emoji = "🟠"
        else:
            assessment = "❌ NEEDS IMPROVEMENT"
            color_emoji = "🔴"
        
        print(f"  Overall Performance: {assessment}")
        print(f"  {color_emoji} MAE: {overall_mae:.3f}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        worst_vital = max(vital_metrics.items(), key=lambda x: x[1]['mae'])
        best_vital = min(vital_metrics.items(), key=lambda x: x[1]['mae'])
        
        print(f"  • Worst performing vital: {worst_vital[0]} (MAE: {worst_vital[1]['mae']:.2f})")
        print(f"  • Best performing vital: {best_vital[0]} (MAE: {best_vital[1]['mae']:.2f})")
        
        if worst_vital[1]['mae'] > 10:
            print(f"  • Consider retraining with more {worst_vital[0]} examples")
        
        worst_scenario = max(scenario_metrics.items(), key=lambda x: x[1]['mae'])
        print(f"  • Most challenging scenario: {worst_scenario[0]} (MAE: {worst_scenario[1]['mae']:.2f})")
        
        return test_report
    
    # ...existing code...

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
    """Main function with performance optimizations"""
    print("🚀 TensorFlow Vital Signs Extractor (Performance Optimized)")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Display optimization status
    print("\n📊 Performance Status:")
    print(f"  CPU threads (inter-op): {tf.config.threading.get_inter_op_parallelism_threads()}")
    print(f"  CPU threads (intra-op): {tf.config.threading.get_intra_op_parallelism_threads()}")
    print(f"  Mixed precision: {'✅ Enabled' if tf.keras.mixed_precision.global_policy().name == 'mixed_float16' else '❌ Disabled'}")
    print(f"  XLA compilation: ✅ Enabled")
    print(f"  GPU available: {'✅ Yes' if tf.config.list_physical_devices('GPU') else '❌ No'}")
    
    extractor = VitalSignsExtractor()
    
    # Check if best model exists
    model_path = "vital_signs_model_best.h5"
    
    # Ask user if they want to run hyperparameter tuning
    print("=== VITAL SIGNS EXTRACTOR ===")
    print("Choose an option:")
    print("1. Run hyperparameter tuning (automatic parameter optimization)")
    print("2. Load existing best model and make predictions")
    print("3. Train with default parameters")
    print("4. Test existing model comprehensively")
    
    try:
        choice = input("Enter your choice (1/2/3/4): ").strip()
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
            try:
                extractor.load_model(model_path)
                print("✅ Best model loaded!")
            except Exception as e:
                print(f"⚠️ Error loading best model: {e}")
                # Try loading from backup
                if extractor.load_model_from_backup(model_path):
                    print("✅ Best model loaded from backup!")
                else:
                    print("❌ No best model found. Running hyperparameter tuning instead...")
                    try:
                        best_params, best_mae = extractor.train_with_hyperparameter_tuning(
                            dataset_dir=None
                        )
                    except Exception as e:
                        print(f"Error: {e}")
                        return
        else:
            print("❌ No best model found. Running hyperparameter tuning instead...")
            try:
                best_params, best_mae = extractor.train_with_hyperparameter_tuning(
                    dataset_dir=None
                )
            except Exception as e:
                print(f"Error: {e}")
                return
    
    elif choice == "3":
        print("Training with default parameters...")
        try:
            history = extractor.train_model(
                dataset_dir=None,
                epochs=50
            )
        except Exception as e:
            print(f"Error training model: {e}")
            return
    
    elif choice == "4":
        # Test existing model
        if os.path.exists(model_path):
            print("Loading best model for testing...")
            try:
                extractor.load_model(model_path)
                print("✅ Model loaded successfully!")
                
                # Ask for test subset size
                try:
                    subset_input = input("Enter test subset size (or press Enter for full dataset): ").strip()
                    test_subset_size = int(subset_input) if subset_input else None
                except:
                    test_subset_size = None
                
                # Run comprehensive test
                print("\n🧪 Starting comprehensive model test...")
                test_report = extractor.comprehensive_model_test(
                    dataset_dir=None,
                    test_subset_size=test_subset_size
                )
                
                if test_report:
                    print(f"\n✅ Model testing completed!")
                    print(f"📊 Overall MAE: {test_report['test_summary']['overall_mae']:.3f}")
                else:
                    print("❌ Model testing failed")
                
                return  # Exit after testing
                
            except Exception as e:
                print(f"❌ Error loading model for testing: {e}")
                return
        else:
            print("❌ No model found for testing. Please train a model first.")
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

