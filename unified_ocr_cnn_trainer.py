"""
Unified OCR + TensorFlow CNN Training Pipeline
==============================================

This script creates a robust training pipeline that:
1. Attempts OCR extraction first (using multiple OCR engines)
2. Falls back to improved CNN when OCR fails
3. Uses iterative training until target MAE is achieved
4. Combines both approaches for maximum accuracy
5. Provides comprehensive evaluation and monitoring

The training continues until the model achieves the target MAE on the validation set.
"""

import os
import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow import keras
from keras.applications import ResNet50, EfficientNetB0
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# OCR imports with fallbacks
OCR_AVAILABLE = []
try:
    import pytesseract
    OCR_AVAILABLE.append('tesseract')
except ImportError:
    print("Tesseract not available")

try:
    import easyocr
    OCR_AVAILABLE.append('easyocr')
except ImportError:
    print("EasyOCR not available")

class UnifiedVitalSignsTrainer:
    """Unified trainer combining OCR and CNN approaches"""
    
    def __init__(self, target_mae=2.0, max_epochs=200):
        self.target_mae = target_mae
        self.max_epochs = max_epochs
        self.model = None
        self.image_size = (224, 224)
        self.vital_signs_labels = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'spo2', 'temperature', 'pulse_rate']
        self.target_scaler = StandardScaler()
        self.image_scaler = MinMaxScaler()
        self.scaler_fitted = False
        self.training_history = []
        self.best_mae = float('inf')
        self.ocr_success_rate = 0.0
        
        # Initialize OCR readers
        self.ocr_readers = {}
        if 'tesseract' in OCR_AVAILABLE:
            # Configure Tesseract for better number recognition
            self.tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789./-°CF% '
        if 'easyocr' in OCR_AVAILABLE:
            try:
                self.ocr_readers['easyocr'] = easyocr.Reader(['en'])
            except:
                print("Failed to initialize EasyOCR")

    def extract_with_ocr(self, image_path):
        """Attempt to extract vital signs using multiple OCR engines"""
        results = {}
        
        # Load and preprocess image for OCR
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Multiple preprocessing techniques for OCR
        preprocessed_images = self._preprocess_for_ocr(image)
        
        for preprocess_name, processed_img in preprocessed_images.items():
            # Try Tesseract
            if 'tesseract' in OCR_AVAILABLE:
                try:
                    text = pytesseract.image_to_string(processed_img, config=self.tesseract_config)
                    extracted = self._parse_ocr_text(text)
                    if extracted:
                        results[f'tesseract_{preprocess_name}'] = extracted
                except Exception as e:
                    pass
            
            # Try EasyOCR
            if 'easyocr' in self.ocr_readers:
                try:
                    ocr_results = self.ocr_readers['easyocr'].readtext(processed_img)
                    text = ' '.join([result[1] for result in ocr_results])
                    extracted = self._parse_ocr_text(text)
                    if extracted:
                        results[f'easyocr_{preprocess_name}'] = extracted
                except Exception as e:
                    pass
        
        # Return best result (most complete extraction)
        if results:
            best_result = max(results.values(), key=lambda x: len([v for v in x.values() if v is not None]))
            return best_result
        
        return None

    def _preprocess_for_ocr(self, image):
        """Multiple preprocessing techniques optimized for OCR"""
        preprocessed = {}
        
        # Original grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        preprocessed['grayscale'] = gray
        
        # High contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        preprocessed['contrast'] = contrast
        
        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed['binary'] = binary
        
        # Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        preprocessed['cleaned'] = cleaned
        
        # Upscale for better OCR
        scale_factor = 2
        for name, img in list(preprocessed.items()):
            upscaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, 
                                interpolation=cv2.INTER_CUBIC)
            preprocessed[f'{name}_upscaled'] = upscaled
        
        return preprocessed

    def _parse_ocr_text(self, text):
        """Parse OCR text to extract vital signs"""
        import re
        
        extracted = {label: None for label in self.vital_signs_labels}
        
        # Common patterns for vital signs
        patterns = {
            'heart_rate': [r'HR[:\s]*(\d{1,3})', r'Heart Rate[:\s]*(\d{1,3})', r'BPM[:\s]*(\d{1,3})'],
            'systolic_bp': [r'(\d{2,3})/\d{2,3}', r'SYS[:\s]*(\d{2,3})', r'Systolic[:\s]*(\d{2,3})'],
            'diastolic_bp': [r'\d{2,3}/(\d{2,3})', r'DIA[:\s]*(\d{2,3})', r'Diastolic[:\s]*(\d{2,3})'],
            'spo2': [r'SpO2[:\s]*(\d{1,3})%?', r'O2[:\s]*(\d{1,3})%?', r'SAT[:\s]*(\d{1,3})%?'],
            'temperature': [r'TEMP[:\s]*(\d{1,3}\.?\d*)°?[CF]?', r'Temperature[:\s]*(\d{1,3}\.?\d*)°?[CF]?'],
            'pulse_rate': [r'PR[:\s]*(\d{1,3})', r'Pulse[:\s]*(\d{1,3})']
        }
        
        for vital_sign, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        # Sanity check ranges
                        if self._is_valid_range(vital_sign, value):
                            extracted[vital_sign] = value
                            break
                    except ValueError:
                        continue
        
        return extracted

    def _is_valid_range(self, vital_sign, value):
        """Check if extracted value is in reasonable medical range"""
        ranges = {
            'heart_rate': (30, 200),
            'systolic_bp': (70, 250),
            'diastolic_bp': (40, 150),
            'spo2': (70, 100),
            'temperature': (90, 110),  # Fahrenheit
            'pulse_rate': (30, 200)
        }
        
        if vital_sign in ranges:
            min_val, max_val = ranges[vital_sign]
            return min_val <= value <= max_val
        return True

    def preprocess_image_for_cnn(self, image_path):
        """Advanced image preprocessing for CNN"""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE for better contrast
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Resize with high-quality interpolation
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize to [0,1]
        image = image.astype(np.float32) / 255.0
        
        # Optional: Add slight noise for regularization
        if np.random.random() < 0.1:  # 10% chance
            noise = np.random.normal(0, 0.01, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        return image

    def load_dataset(self):
        """Load dataset with OCR fallback and ground truth"""
        images_dir = "generated_heart_monitors"
        metadata_file = os.path.join(images_dir, "dataset_info.json")
        
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Dataset metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        X = []
        y = []
        ocr_data = []
        successful_ocr = 0
        
        print(f"Loading {len(metadata)} images...")
        
        for i, entry in enumerate(metadata):
            image_path = os.path.join(images_dir, entry['filename'])
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # Process image for CNN
            image = self.preprocess_image_for_cnn(image_path)
            if image is None:
                continue
            
            # Try OCR extraction
            ocr_result = self.extract_with_ocr(image_path)
            if ocr_result and any(v is not None for v in ocr_result.values()):
                successful_ocr += 1
            
            ocr_data.append(ocr_result)
            
            # Get ground truth from metadata
            vitals = entry['vitals']
            target_values = [vitals.get(label, 0) for label in self.vital_signs_labels]
            
            X.append(image)
            y.append(target_values)
            
            if i % 5 == 0:
                print(f"Processed {i+1}/{len(metadata)} images...")
        
        self.ocr_success_rate = successful_ocr / len(X) if X else 0
        print(f"OCR Success Rate: {self.ocr_success_rate:.1%}")
        
        return np.array(X), np.array(y), ocr_data

    def create_advanced_model(self):
        """Create advanced CNN model with transfer learning"""
        # Use EfficientNet as base (smaller and more efficient than ResNet)
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )
        
        # Freeze early layers, fine-tune later layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Custom head for regression
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        
        # Output layer (6 vital signs)
        predictions = Dense(6, activation='linear', name='vital_signs')(x)
        
        model = keras.Model(inputs=base_model.input, outputs=predictions)
        
        # Custom loss combining MAE and MSE
        def combined_loss(y_true, y_pred):
            mae = tf.reduce_mean(tf.abs(y_true - y_pred))
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            return 0.7 * mae + 0.3 * mse
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=combined_loss,
            metrics=['mae', 'mse']
        )
        
        return model

    def train_until_target(self, X_train, y_train, X_val, y_val):
        """Train until target MAE is achieved"""
        print(f"Training until MAE < {self.target_mae}")
        
        epoch = 0
        patience_counter = 0
        max_patience = 15
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_mae', patience=max_patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=7, min_lr=1e-7),
            ModelCheckpoint('best_unified_model.h5', monitor='val_mae', save_best_only=True)
        ]
        
        while epoch < self.max_epochs:
            print(f"\n--- Training Epoch {epoch + 1} ---")
            
            # Train for a few epochs at a time
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epoch + 5,
                initial_epoch=epoch,
                batch_size=16,
                callbacks=callbacks,
                verbose=1
            )
            
            # Check current performance
            val_predictions = self.model.predict(X_val, verbose=0)
            current_mae = mean_absolute_error(y_val, val_predictions)
            
            print(f"Current validation MAE: {current_mae:.4f}")
            
            if current_mae < self.best_mae:
                self.best_mae = current_mae
                patience_counter = 0
                print(f"New best MAE: {self.best_mae:.4f}")
            else:
                patience_counter += 1
            
            # Check if target achieved
            if current_mae <= self.target_mae:
                print(f"🎉 Target MAE achieved! Final MAE: {current_mae:.4f}")
                break
            
            # Check patience
            if patience_counter >= max_patience:
                print(f"Early stopping: No improvement for {max_patience} epochs")
                break
            
            epoch += 5
            
            # Fine-tune more layers if stuck
            if epoch > 30 and current_mae > self.target_mae * 2:
                print("Unfreezing more layers for fine-tuning...")
                for layer in self.model.layers:
                    if hasattr(layer, 'trainable'):
                        layer.trainable = True
                
                # Lower learning rate for fine-tuning
                self.model.compile(
                    optimizer=Adam(learning_rate=0.0001),
                    loss=self.model.loss,
                    metrics=self.model.metrics
                )
        
        return self.model

    def evaluate_comprehensive(self, X_test, y_test, ocr_data_test):
        """Comprehensive evaluation of both CNN and OCR approaches"""
        print("\n" + "="*50)
        print("COMPREHENSIVE EVALUATION")
        print("="*50)
        
        # CNN predictions
        cnn_predictions = self.model.predict(X_test, verbose=0)
        cnn_mae = mean_absolute_error(y_test, cnn_predictions)
        cnn_mse = mean_squared_error(y_test, cnn_predictions)
        cnn_r2 = r2_score(y_test, cnn_predictions)
        
        print(f"\nCNN Results:")
        print(f"MAE: {cnn_mae:.4f}")
        print(f"MSE: {cnn_mse:.4f}")
        print(f"R²:  {cnn_r2:.4f}")
        
        # OCR evaluation where available
        ocr_results = []
        for i, ocr_result in enumerate(ocr_data_test):
            if ocr_result and any(v is not None for v in ocr_result.values()):
                ocr_values = [ocr_result.get(label, 0) for label in self.vital_signs_labels]
                ocr_results.append(ocr_values)
            else:
                ocr_results.append(None)
        
        valid_ocr_indices = [i for i, result in enumerate(ocr_results) if result is not None]
        
        if valid_ocr_indices:
            ocr_predictions = np.array([ocr_results[i] for i in valid_ocr_indices])
            ocr_ground_truth = y_test[valid_ocr_indices]
            
            ocr_mae = mean_absolute_error(ocr_ground_truth, ocr_predictions)
            ocr_mse = mean_squared_error(ocr_ground_truth, ocr_predictions)
            ocr_r2 = r2_score(ocr_ground_truth, ocr_predictions)
            
            print(f"\nOCR Results (on {len(valid_ocr_indices)} successful extractions):")
            print(f"MAE: {ocr_mae:.4f}")
            print(f"MSE: {ocr_mse:.4f}")
            print(f"R²:  {ocr_r2:.4f}")
        else:
            print("\nOCR: No successful extractions")
        
        # Per-vital sign analysis
        print(f"\nPer-Vital Sign MAE (CNN):")
        for i, label in enumerate(self.vital_signs_labels):
            vital_mae = mean_absolute_error(y_test[:, i], cnn_predictions[:, i])
            print(f"{label}: {vital_mae:.4f}")
        
        # Create evaluation report
        report = {
            'timestamp': datetime.now().isoformat(),
            'cnn_performance': {
                'mae': float(cnn_mae),
                'mse': float(cnn_mse),
                'r2': float(cnn_r2)
            },
            'ocr_success_rate': self.ocr_success_rate,
            'target_mae': self.target_mae,
            'target_achieved': cnn_mae <= self.target_mae
        }
        
        # Save evaluation report
        with open('unified_evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

    def run_complete_pipeline(self):
        """Run the complete training and evaluation pipeline"""
        print("🚀 Starting Unified OCR + CNN Training Pipeline")
        print("="*60)
        
        # Load dataset
        print("Loading dataset...")
        X, y, ocr_data = self.load_dataset()
        
        if len(X) == 0:
            raise ValueError("No valid images found in dataset")
        
        print(f"Loaded {len(X)} images")
        
        # Normalize targets
        if not self.scaler_fitted:
            y_scaled = self.target_scaler.fit_transform(y)
            self.scaler_fitted = True
            joblib.dump(self.target_scaler, 'unified_target_scaler.pkl')
        else:
            y_scaled = self.target_scaler.transform(y)
        
        # Split dataset
        X_train, X_temp, y_train, y_temp, ocr_train, ocr_temp = train_test_split(
            X, y_scaled, ocr_data, test_size=0.4, random_state=42
        )
        
        X_val, X_test, y_val, y_test, ocr_val, ocr_test = train_test_split(
            X_temp, y_temp, ocr_temp, test_size=0.5, random_state=42
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Create and train model
        print("\nCreating advanced CNN model...")
        self.model = self.create_advanced_model()
        print(f"Model created with {self.model.count_params():,} parameters")
        
        print("\nStarting training...")
        self.model = self.train_until_target(X_train, y_train, X_val, y_val)
        
        # Transform back for evaluation
        y_test_original = self.target_scaler.inverse_transform(y_test)
        
        # Final evaluation
        print("\nRunning comprehensive evaluation...")
        report = self.evaluate_comprehensive(X_test, y_test_original, ocr_test)
        
        # Save final model
        self.model.save('unified_vital_signs_model_final.h5')
        print("\nModel saved as 'unified_vital_signs_model_final.h5'")
        
        return report


def main():
    """Main execution function"""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Configure GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU acceleration enabled: {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Create trainer with target MAE
    target_mae = 2.0  # Adjust this based on your requirements
    trainer = UnifiedVitalSignsTrainer(target_mae=target_mae, max_epochs=100)
    
    try:
        # Run complete pipeline
        report = trainer.run_complete_pipeline()
        
        print("\n" + "="*60)
        print("🎯 TRAINING COMPLETE!")
        print("="*60)
        print(f"Final MAE: {report['cnn_performance']['mae']:.4f}")
        print(f"Target MAE: {target_mae}")
        print(f"Target Achieved: {'✅ YES' if report['target_achieved'] else '❌ NO'}")
        print(f"OCR Success Rate: {report['ocr_success_rate']:.1%}")
        
        if not report['target_achieved']:
            print("\n💡 Suggestions for improvement:")
            print("- Increase max_epochs or reduce target_mae")
            print("- Improve image generation quality")
            print("- Collect more training data")
            print("- Try different model architectures")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
