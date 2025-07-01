"""
Test and compare the hybrid OCR+CNN approach with the original metadata approach
"""

import os
import json
import numpy as np
from sklearn.metrics import mean_absolute_error

def compare_approaches(dataset_dir):
    """Compare OCR ground truth vs metadata ground truth"""
    
    print("🔍 COMPARING OCR vs METADATA GROUND TRUTH")
    print("=" * 60)
    
    # Load OCR results if available
    ocr_file = 'ocr_ground_truth.json'
    if not os.path.exists(ocr_file):
        print(f"❌ {ocr_file} not found. Run hybrid_ocr_cnn_trainer.py first.")
        return
    
    with open(ocr_file, 'r') as f:
        ocr_results = json.load(f)
    
    # Load metadata
    metadata_path = os.path.join(dataset_dir, "dataset_info.json")
    with open(metadata_path, 'r') as f:
        dataset_info = json.load(f)
    
    # Create metadata lookup
    metadata_lookup = {item['filename']: item['vitals'] for item in dataset_info}
    
    # Compare OCR vs Metadata
    vital_labels = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'spo2', 'temperature', 'pulse_rate']
    
    comparisons = []
    for result in ocr_results:
        filename = result['filename']
        ocr_vitals = result['ocr_vitals']
        metadata_vitals = metadata_lookup.get(filename, {})
        
        comparison = {
            'filename': filename,
            'differences': {}
        }
        
        for vital in vital_labels:
            if vital in ocr_vitals and vital in metadata_vitals:
                ocr_val = ocr_vitals[vital]
                meta_val = metadata_vitals[vital]
                diff = abs(ocr_val - meta_val)
                comparison['differences'][vital] = {
                    'ocr': ocr_val,
                    'metadata': meta_val,
                    'difference': diff
                }
        
        comparisons.append(comparison)
    
    # Analyze differences
    print(f"📊 ANALYSIS OF {len(comparisons)} IMAGES:")
    print("-" * 40)
    
    all_diffs = {vital: [] for vital in vital_labels}
    
    for comp in comparisons:
        for vital, data in comp['differences'].items():
            all_diffs[vital].append(data['difference'])
    
    # Summary statistics
    for vital in vital_labels:
        if all_diffs[vital]:
            mean_diff = np.mean(all_diffs[vital])
            max_diff = np.max(all_diffs[vital])
            matches = sum(1 for d in all_diffs[vital] if d < 1.0)
            total = len(all_diffs[vital])
            
            print(f"{vital:<15}: Mean Diff={mean_diff:6.2f} | Max Diff={max_diff:6.1f} | Matches={matches}/{total}")
    
    # Show examples of differences
    print(f"\n📋 EXAMPLES OF DIFFERENCES:")
    print("-" * 50)
    
    for i, comp in enumerate(comparisons[:5]):
        print(f"\n{comp['filename']}:")
        for vital, data in comp['differences'].items():
            if data['difference'] > 0:
                print(f"  {vital}: OCR={data['ocr']} vs Meta={data['metadata']} (diff={data['difference']:.1f})")
    
    # Determine which approach to use
    total_avg_diff = np.mean([np.mean(diffs) for diffs in all_diffs.values() if diffs])
    
    print(f"\n🎯 RECOMMENDATION:")
    print("-" * 30)
    
    if total_avg_diff < 5.0:
        print(f"✅ OCR and metadata are reasonably consistent (avg diff: {total_avg_diff:.2f})")
        print("   → Use hybrid OCR+CNN approach for best results")
    else:
        print(f"⚠️ Large differences between OCR and metadata (avg diff: {total_avg_diff:.2f})")
        print("   → Check image generation - OCR might be reading different values")
        print("   → Consider using pure OCR approach instead of metadata")

def test_trained_model():
    """Test the trained hybrid model"""
    
    model_file = 'hybrid_ocr_cnn_best.keras'
    if not os.path.exists(model_file):
        print(f"❌ {model_file} not found. Train the model first.")
        return
    
    print(f"\n🧪 TESTING TRAINED HYBRID MODEL")
    print("=" * 40)
    
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_file)
        print(f"✅ Model loaded successfully")
        print(f"📋 Model summary:")
        model.summary()
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated_heart_monitors'))
    
    # Compare approaches
    compare_approaches(dataset_dir)
    
    # Test trained model
    test_trained_model()
