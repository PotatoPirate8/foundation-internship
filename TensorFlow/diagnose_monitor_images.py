"""
Diagnostic script to analyze what's actually visible in the monitor images
"""

import os
import cv2
import numpy as np
import json

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

def analyze_single_image(image_path, show_debug=True):
    """Analyze a single image in detail"""
    
    print(f"\n🔍 ANALYZING: {os.path.basename(image_path)}")
    print("-" * 50)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Failed to load image")
        return
    
    height, width = image.shape[:2]
    print(f"📐 Image size: {width}x{height}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Show basic image statistics
    print(f"📊 Pixel stats - Min: {gray.min()}, Max: {gray.max()}, Mean: {gray.mean():.1f}")
    
    # Check if image is mostly uniform (synthetic)
    std_dev = gray.std()
    print(f"📈 Standard deviation: {std_dev:.2f}")
    
    if std_dev < 10:
        print("⚠️ Very low variation - likely synthetic/uniform image")
    
    # Try different preprocessing techniques
    preprocessing_methods = {
        "Original": gray,
        "CLAHE": cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray),
        "Binary": cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        "Inverted": cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        "Adaptive": cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    }
    
    if OCR_AVAILABLE:
        print(f"\n🔤 OCR RESULTS:")
        print("-" * 30)
        
        for method_name, processed_img in preprocessing_methods.items():
            try:
                # Try different OCR configurations
                configs = [
                    r'--oem 3 --psm 6',  # Default
                    r'--oem 3 --psm 7',  # Single text line
                    r'--oem 3 --psm 8',  # Single word
                    r'--oem 3 --psm 13', # Raw line
                    r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789./'  # Numbers only
                ]
                
                best_text = ""
                for config in configs:
                    try:
                        text = pytesseract.image_to_string(processed_img, config=config).strip()
                        if len(text) > len(best_text):
                            best_text = text
                    except:
                        continue
                
                if best_text:
                    print(f"  {method_name:<12}: '{best_text}'")
                else:
                    print(f"  {method_name:<12}: (no text detected)")
                
            except Exception as e:
                print(f"  {method_name:<12}: OCR error - {e}")
    
    else:
        print("⚠️ OCR not available")
    
    # Save debug images if requested
    if show_debug:
        debug_dir = "debug_images"
        os.makedirs(debug_dir, exist_ok=True)
        
        filename_base = os.path.splitext(os.path.basename(image_path))[0]
        
        for method_name, processed_img in preprocessing_methods.items():
            debug_path = os.path.join(debug_dir, f"{filename_base}_{method_name.lower()}.png")
            cv2.imwrite(debug_path, processed_img)
        
        print(f"💾 Debug images saved to {debug_dir}/")

def check_image_generation_code():
    """Check if we can find the image generation code"""
    
    print(f"\n🔍 LOOKING FOR IMAGE GENERATION CODE")
    print("-" * 40)
    
    # Look for Python files that might generate images
    search_paths = [
        "..",
        "../..",
        "."
    ]
    
    potential_files = []
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith('.py') and any(keyword in file.lower() for keyword in ['generate', 'monitor', 'image', 'create']):
                        potential_files.append(os.path.join(root, file))
    
    if potential_files:
        print("📁 Found potential image generation files:")
        for file in potential_files[:10]:  # Show first 10
            print(f"  {file}")
    else:
        print("❌ No image generation files found")

def main():
    """Main diagnostic function"""
    
    print("🩺 MEDICAL MONITOR IMAGE DIAGNOSTIC")
    print("=" * 60)
    
    # Set dataset directory
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated_heart_monitors'))
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    # Load metadata
    metadata_path = os.path.join(dataset_dir, "dataset_info.json")
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata not found: {metadata_path}")
        return
    
    with open(metadata_path, 'r') as f:
        dataset_info = json.load(f)
    
    print(f"📊 Found {len(dataset_info)} images in dataset")
    
    # Analyze first few images in detail
    for i in range(min(3, len(dataset_info))):
        item = dataset_info[i]
        image_path = os.path.join(dataset_dir, item['filename'])
        
        if os.path.exists(image_path):
            print(f"\n📋 METADATA FOR {item['filename']}:")
            for vital, value in item['vitals'].items():
                print(f"  {vital}: {value}")
            
            analyze_single_image(image_path, show_debug=(i == 0))  # Only save debug for first image
    
    # Check for image generation code
    check_image_generation_code()
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 30)
    print("1. Check the debug_images/ folder to see processed versions")
    print("2. If images are blank/uniform, the generation code needs fixing")
    print("3. If OCR finds no text, images might be purely graphical (no text)")
    print("4. Consider using the original CNN approach with proper normalization")
    print("5. Or fix the image generation to include readable text")

if __name__ == "__main__":
    main()
