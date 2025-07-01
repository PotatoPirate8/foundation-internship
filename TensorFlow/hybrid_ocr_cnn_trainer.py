"""
Hybrid OCR + TensorFlow Training Loop
=====================================

This script uses OCR to extract the actual values displayed in the monitor images,
then trains a CNN model to predict these OCR-extracted values (not the metadata ground truth).

This solves the fundamental issue where the model was trying to learn a relationship
that might not exist between the image pixels and the metadata ground truth.
"""

import os
import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib
import re
from datetime import datetime

# Import OCR functionality
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    print("⚠️ pytesseract not available. Install with: pip install pytesseract")
    OCR_AVAILABLE = False

class OCRGroundTruthExtractor:
    """Extract ground truth values from images using OCR"""
    
    def __init__(self):
        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789./'
        self.vital_ranges = {
            'heart_rate': (30, 250),
            'systolic_bp': (60, 250),
            'diastolic_bp': (30, 150),
            'spo2': (70, 100),
            'temperature': (30.0, 45.0),
            'pulse_rate': (30, 250)
        }

    def preprocess_for_ocr(self, image):
        """Create multiple preprocessed versions for better OCR"""
        processed_images = []
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Original grayscale
        processed_images.append(gray)
        
        # 2. High contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        processed_images.append(contrast)
        
        # 3. Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(binary)
        
        # 4. Inverted binary
        inverted = cv2.bitwise_not(binary)
        processed_images.append(inverted)
        
        return processed_images

    def extract_vitals_from_image(self, image_path):
        """Extract vital signs from image using OCR"""
        if not OCR_AVAILABLE:
            return None
            
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        processed_images = self.preprocess_for_ocr(image)
        all_numbers = []
        
        # Extract text from all preprocessed versions
        for proc_img in processed_images:
            try:
                text = pytesseract.image_to_string(proc_img, config=self.tesseract_config)
                numbers = re.findall(r'\d+(?:\.\d+)?', text)
                all_numbers.extend([float(num) for num in numbers])
            except:
                continue
        
        if not all_numbers:
            return None
        
        # Parse vital signs using ranges and patterns
        vitals = {}
        
        # Blood pressure pattern (X/Y)
        bp_pattern = r'(\d+)/(\d+)'
        for proc_img in processed_images:
            try:
                text = pytesseract.image_to_string(proc_img, config=self.tesseract_config)
                bp_matches = re.findall(bp_pattern, text)
                for match in bp_matches:
                    systolic, diastolic = int(match[0]), int(match[1])
                    if (self.vital_ranges['systolic_bp'][0] <= systolic <= self.vital_ranges['systolic_bp'][1] and
                        self.vital_ranges['diastolic_bp'][0] <= diastolic <= self.vital_ranges['diastolic_bp'][1]):
                        vitals['systolic_bp'] = float(systolic)
                        vitals['diastolic_bp'] = float(diastolic)
                        break
                if 'systolic_bp' in vitals:
                    break
            except:
                continue
        
        # Extract other vitals by range
        unique_numbers = list(set(all_numbers))
        
        # Heart rate (30-250)
        hr_candidates = [n for n in unique_numbers if self.vital_ranges['heart_rate'][0] <= n <= self.vital_ranges['heart_rate'][1]]
        if hr_candidates:
            vitals['heart_rate'] = hr_candidates[0]
            vitals['pulse_rate'] = hr_candidates[0]  # Often the same
        
        # SpO2 (70-100)
        spo2_candidates = [n for n in unique_numbers if self.vital_ranges['spo2'][0] <= n <= self.vital_ranges['spo2'][1]]
        if spo2_candidates:
            vitals['spo2'] = max(spo2_candidates)
        
        # Temperature (30-45)
        temp_candidates = [n for n in unique_numbers if self.vital_ranges['temperature'][0] <= n <= self.vital_ranges['temperature'][1]]
        if temp_candidates:
            vitals['temperature'] = temp_candidates[0]
        
        return vitals

class HybridCNNTrainer:
    """CNN trainer that uses OCR-extracted ground truth"""
    
    def __init__(self):
        self.model = None
        self.image_size = (224, 224)
        self.vital_signs_labels = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'spo2', 'temperature', 'pulse_rate']
        self.target_scaler = StandardScaler()
        self.scaler_fitted = False
        self.ocr_extractor = OCRGroundTruthExtractor()

    def preprocess_image(self, image_path):
        """Preprocess image for CNN"""
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
        
        # Resize
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        return image

    def extract_ocr_ground_truth(self, dataset_dir):
        """Extract ground truth using OCR from all images"""
        metadata_path = os.path.join(dataset_dir, "dataset_info.json")
        with open(metadata_path, 'r') as f:
            dataset_info = json.load(f)
        
        print("🔍 Extracting OCR ground truth from images...")
        
        ocr_results = []
        successful_extractions = 0
        
        for i, item in enumerate(dataset_info):
            image_path = os.path.join(dataset_dir, item['filename'])
            
            if os.path.exists(image_path):
                # Extract vitals using OCR
                ocr_vitals = self.ocr_extractor.extract_vitals_from_image(image_path)
                
                if ocr_vitals and len(ocr_vitals) >= 4:  # Need at least 4 vital signs
                    # Create complete vital vector (fill missing with NaN)
                    vital_vector = []
                    for vital in self.vital_signs_labels:
                        if vital in ocr_vitals:
                            vital_vector.append(ocr_vitals[vital])
                        else:
                            # Use metadata as fallback for missing values
                            vital_vector.append(item['vitals'].get(vital, np.nan))
                    
                    ocr_results.append({
                        'filename': item['filename'],
                        'image_path': image_path,
                        'ocr_vitals': ocr_vitals,
                        'vital_vector': vital_vector,
                        'metadata_vitals': item['vitals']
                    })
                    successful_extractions += 1
                
                if (i + 1) % 5 == 0:
                    print(f"  Processed {i + 1}/{len(dataset_info)} images... (Success: {successful_extractions})")
        
        print(f"✅ Successfully extracted OCR ground truth from {successful_extractions}/{len(dataset_info)} images")
        
        # Save OCR results for analysis
        with open('ocr_ground_truth.json', 'w') as f:
            json.dump(ocr_results, f, indent=2, default=str)
        
        return ocr_results

    def create_cnn_model(self):
        """Create CNN model with transfer learning"""
        # Use EfficientNet as backbone
        base_model = keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Build model
        inputs = keras.Input(shape=(*self.image_size, 3))
        x = keras.applications.efficientnet.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = keras.layers.Dense(6, activation='linear')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='huber',
            metrics=['mae']
        )
        
        return model

    def prepare_training_data(self, ocr_results):
        """Prepare training data from OCR results"""
        images = []
        labels = []
        
        print("📦 Preparing training data...")
        
        for result in ocr_results:
            # Load and preprocess image
            image = self.preprocess_image(result['image_path'])
            if image is not None:
                images.append(image)
                labels.append(result['vital_vector'])
        
        images = np.array(images)
        labels = np.array(labels)
        
        # Remove samples with NaN values
        valid_mask = ~np.isnan(labels).any(axis=1)
        images = images[valid_mask]
        labels = labels[valid_mask]
        
        print(f"📊 Training data prepared: {len(images)} samples")
        
        # Normalize labels
        if not self.scaler_fitted:
            self.target_scaler.fit(labels)
            self.scaler_fitted = True
            joblib.dump(self.target_scaler, 'ocr_target_scaler.pkl')
        
        normalized_labels = self.target_scaler.transform(labels)
        
        return images, normalized_labels, labels

    def train_with_ocr_ground_truth(self, dataset_dir, epochs=100, target_mae=2.0):
        """Main training loop using OCR ground truth"""
        print("🚀 HYBRID OCR + CNN TRAINING")
        print("=" * 50)
        
        # Step 1: Extract OCR ground truth
        ocr_results = self.extract_ocr_ground_truth(dataset_dir)
        
        if len(ocr_results) < 10:
            print("❌ Not enough successful OCR extractions for training")
            return None
        
        # Step 2: Prepare training data
        X, y_norm, y_orig = self.prepare_training_data(ocr_results)
        
        # Step 3: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_norm, test_size=0.2, random_state=42
        )
        
        print(f"📈 Training set: {len(X_train)} samples")
        print(f"📉 Test set: {len(X_test)} samples")
        
        # Step 4: Create and train model
        self.model = self.create_cnn_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_mae', patience=15, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_mae', factor=0.5, patience=8, min_lr=1e-7, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'hybrid_ocr_cnn_best.keras', 
                monitor='val_mae', save_best_only=True, verbose=1
            )
        ]
        
        # Training loop
        print(f"\n🏋️ Training model (target MAE: {target_mae})...")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=8,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n📊 Final Test MAE: {test_mae:.4f}")
        
        # Denormalize and test predictions
        self.evaluate_predictions(X_test, y_test, ocr_results[-len(X_test):])
        
        return history

    def evaluate_predictions(self, X_test, y_test_norm, test_ocr_results):
        """Evaluate predictions against OCR ground truth"""
        print(f"\n🔍 EVALUATING PREDICTIONS")
        print("-" * 40)
        
        # Make predictions
        predictions_norm = self.model.predict(X_test, verbose=0)
        
        # Denormalize
        predictions = self.target_scaler.inverse_transform(predictions_norm)
        y_test = self.target_scaler.inverse_transform(y_test_norm)
        
        # Calculate per-vital MAE
        for i, vital in enumerate(self.vital_signs_labels):
            mae = mean_absolute_error(y_test[:, i], predictions[:, i])
            print(f"{vital:<15}: MAE = {mae:6.2f}")
        
        # Show sample predictions
        print(f"\n📋 SAMPLE PREDICTIONS:")
        print("-" * 60)
        
        for i in range(min(5, len(predictions))):
            if i < len(test_ocr_results):
                filename = os.path.basename(test_ocr_results[i]['image_path'])
                print(f"\nImage: {filename}")
                print("Predicted vs OCR Ground Truth:")
                
                for j, vital in enumerate(self.vital_signs_labels):
                    pred_val = predictions[i][j]
                    true_val = y_test[i][j]
                    error = abs(pred_val - true_val)
                    
                    if vital == 'temperature':
                        print(f"  {vital:<12}: {pred_val:6.1f} vs {true_val:6.1f} (error: {error:5.1f})")
                    else:
                        print(f"  {vital:<12}: {pred_val:6.0f} vs {true_val:6.0f} (error: {error:5.0f})")

def main():
    """Main execution function"""
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated_heart_monitors'))
    
    if not OCR_AVAILABLE:
        print("❌ OCR not available. Please install pytesseract first.")
        return
    
    # Create trainer
    trainer = HybridCNNTrainer()
    
    # Train model
    history = trainer.train_with_ocr_ground_truth(
        dataset_dir=dataset_dir,
        epochs=100,
        target_mae=2.0
    )
    
    if history:
        print(f"\n✅ Training completed! Model saved as 'hybrid_ocr_cnn_best.keras'")
        print(f"📄 OCR ground truth saved as 'ocr_ground_truth.json'")
    else:
        print(f"\n❌ Training failed!")

if __name__ == "__main__":
    main()
