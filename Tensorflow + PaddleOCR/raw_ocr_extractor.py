"""
Raw OCR Data Extractor - Returns unprocessed PaddleOCR results

This script extracts raw OCR data from images using PaddleOCR without any 
processing or interpretation. It returns the original data structure from PaddleOCR.
"""

from paddleocr import PaddleOCR
import warnings
import os
import json
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Global OCR instance for model reuse
_ocr_instance = None

def get_ocr_instance():
    """Get or create a global OCR instance (singleton pattern)"""
    global _ocr_instance
    if _ocr_instance is None:
        print("🔧 Initializing PaddleOCR...")
        
        # Set environment variables for model caching
        cache_dir = os.path.expanduser('~/.paddlex')
        os.environ['PADDLEHUB_HOME'] = cache_dir
        os.environ['HUB_HOME'] = cache_dir
        os.environ['PADDLE_INSTALL_DIR'] = cache_dir
        
        # Check if models are cached
        model_cache_dir = os.path.join(cache_dir, 'official_models')
        if os.path.exists(model_cache_dir):
            print(f"📁 Using cached models from: {model_cache_dir}")
        
        # Initialize using your generic pattern
        _ocr_instance = PaddleOCR(
            use_doc_orientation_classify=False, 
            use_doc_unwarping=False, 
            use_textline_orientation=False
        )
        print("✅ PaddleOCR initialized successfully!")
    else:
        print("♻️ Reusing existing PaddleOCR instance")
    return _ocr_instance

def extract_raw_ocr_data(image_path, save_files=True, output_dir="output"):
    """
    Extract raw OCR data from an image
    
    Args:
        image_path: Path to the image file
        save_files: Whether to save output files using PaddleOCR's methods
        output_dir: Directory to save output files
        
    Returns:
        Raw PaddleOCR result structure
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"🔍 Processing image: {os.path.basename(image_path)}")
    
    ocr = get_ocr_instance()
    result = ocr.predict(image_path)
    
    print(f"\n📊 Raw OCR Results:")
    print(f"   Result type: {type(result)}")
    print(f"   Number of results: {len(result) if result else 0}")
    
    if result and len(result) > 0:
        for i, res in enumerate(result):
            print(f"\n🔍 Result {i}:")
            print(f"   Type: {type(res)}")
            
            # Print the raw result using PaddleOCR's print method
            print("   Raw content:")
            res.print()
            
            if save_files:
                try:
                    # Create output directory if it doesn't exist
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    # Save using PaddleOCR's methods
                    res.save_to_img(output_dir)
                    res.save_to_json(output_dir)
                    print(f"   ✅ Files saved to {output_dir}/")
                    
                except Exception as e:
                    print(f"   ❌ Error saving files: {e}")
    
    return result

def save_raw_data_to_json(raw_result, output_file):
    """
    Save raw OCR data to a JSON file for further analysis
    
    Args:
        raw_result: Raw PaddleOCR result
        output_file: Output JSON file path
    """
    if not raw_result:
        return
    
    # Extract the raw data structure
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "result_type": str(type(raw_result)),
        "result_count": len(raw_result) if raw_result else 0,
        "raw_results": []
    }
    
    for i, res in enumerate(raw_result):
        if hasattr(res, 'res'):
            # Extract the main result dictionary
            result_dict = dict(res.res)
            
            # Convert numpy arrays to lists for JSON serialization
            if 'dt_polys' in result_dict:
                result_dict['dt_polys'] = result_dict['dt_polys'].tolist()
            if 'rec_polys' in result_dict:
                result_dict['rec_polys'] = result_dict['rec_polys'].tolist()
            if 'rec_boxes' in result_dict:
                result_dict['rec_boxes'] = result_dict['rec_boxes'].tolist()
            if 'rec_scores' in result_dict:
                result_dict['rec_scores'] = result_dict['rec_scores'].tolist()
            if 'textline_orientation_angles' in result_dict:
                result_dict['textline_orientation_angles'] = result_dict['textline_orientation_angles'].tolist()
            
            output_data["raw_results"].append({
                "result_index": i,
                "result_data": result_dict
            })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"📁 Raw data saved to: {output_file}")

if __name__ == "__main__":
    # Test with the monitor image
    image_path = "../generated_heart_monitors/monitor_001_low_spo2.png"
    
    if os.path.exists(image_path):
        print("🚀 Extracting raw OCR data...")
        raw_result = extract_raw_ocr_data(image_path, save_files=True, output_dir="raw_output")
        
        # Save detailed JSON data
        save_raw_data_to_json(raw_result, "complete_raw_ocr_data.json")
        
        print(f"\n✅ Raw OCR extraction complete!")
        print(f"   Check 'raw_output/' folder for images and basic JSON")
        print(f"   Check 'complete_raw_ocr_data.json' for detailed raw data")
        
    else:
        print(f"❌ Test image not found: {image_path}")
        # List available images
        print("\n📂 Available images in generated_heart_monitors:")
        try:
            monitors_dir = "../generated_heart_monitors"
            if os.path.exists(monitors_dir):
                images = [f for f in os.listdir(monitors_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                for img in images[:5]:  # Show first 5
                    print(f"   - {img}")
        except Exception as e:
            print(f"   Error listing images: {e}")
