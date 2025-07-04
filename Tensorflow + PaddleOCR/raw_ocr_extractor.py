"""
Raw OCR Data Extractor - Returns unprocessed PaddleOCR results

This script extracts raw OCR data from images using PaddleOCR without any 
processing or interpretation. It returns the original data structure from PaddleOCR.
"""
import pprint

from botocore.hooks import first_non_none_response
from paddleocr import PaddleOCR
import os
import json
from datetime import datetime

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

    if not result or not isinstance(result, list):
        raise ValueError("Invalid OCR result format")

    # Debug
    pprint.pprint(result, depth=2)

    print(f"\n📊 Raw OCR Results:")
    print(f"   Result type: {type(result)}")
    print(f"   Number of results: {len(result) if result else 0}")

    if len(result) > 0:
        first_result = result[0]


        # Try extract coordinate data for future sorting
        if 'rec_boxes' in first_result and 'rec_texts' in first_result:
            boxes = first_result['rec_boxes']
            texts = first_result['rec_texts']

            print(f"\n📊 Found {len(boxes)} text elements:")
            for i, (box, text) in enumerate(zip(boxes, texts)):
                x1, y1, x2, y2 = box
                print(f"  {i}: '{text}' at [{x1},{y1},{x2},{y2}]")

        for i, res in enumerate(result):
            print(f"\n🔍 Result {i}:")
            print(f"   Type: {type(res)}")

            # if hasattr(res, 'bbox'):  # Try extract coordinate data for further sorting
            #     bbox = res.bbox
            #     print(f" Coordinates for (x1,y1,x2,y2): [{bbox[0][0]}], [{bbox[0][1]}], {bbox[2][0]}], [{bbox[2][1]}]")
            #     print(f"   Full bbox (4 corners): {bbox}")  # Shows all 4 points

            # elif hasattr(res, 'points'): # In case we are using diff PaddleOCR version
            #     points = res.points
            #     x_coords = [p[0] for p in points]
            #     y_coords = [p[1] for p in points]
            #     x1, y1 = min(x_coords), min(y_coords)
            #     x2, y2 = max(x_coords), max(y_coords)
            #     print(f"   Coordinates (x1,y1,x2,y2): [{x1}, {y1}, {x2}, {y2}]")
            #     print(f"   Polygon points: {points}")

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


def save_raw_data_to_json(raw_result, output_dir="output"):
    """
    Save raw OCR data to JSON using PaddleOCR's built-in method
    
    Args:
        raw_result: Raw PaddleOCR result
        output_dir: Directory to save JSON files
    """
    if not raw_result or not isinstance(raw_result, list):
        print("❌ No raw result to save")
        return



    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"� Saving raw OCR data to JSON...")

    for i, res in enumerate(raw_result):
        try:
            # print(f"   Result: {res}")
            res.save_to_json(output_dir)
            print(f"   ✅ Result {i} saved to JSON in {output_dir}/")
        except Exception as e:
            print(f"   ❌ Error saving result {i}: {e}")

    print(f"� All raw OCR data saved to: {output_dir}/")


if __name__ == "__main__":
    # Test with the monitor image
    image_path = "../Heart monitor images/medical_monitor.jpg"

    if os.path.exists(image_path):
        print("🚀 Extracting raw OCR data...")
        raw_result = extract_raw_ocr_data(image_path, save_files=True, output_dir="raw_output")

        # Save detailed JSON data using PaddleOCR's built-in method
        save_raw_data_to_json(raw_result, "complete_raw_ocr_data")

        print(f"\n✅ Raw OCR extraction complete!")
        print(f"   Check 'raw_output/' folder for images and basic JSON")
        print(f"   Check 'complete_raw_ocr_data/' folder for detailed raw JSON data")

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
