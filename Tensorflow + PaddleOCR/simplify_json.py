import json
import os
from datetime import datetime

def extract_simplified_ocr_data(input_json_path):
    """Extract rec_texts, rec_scores and important metadata from raw OCR JSON"""
    
    # Read the raw OCR data
    with open(input_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Extract important information
    simplified_data = {
        "timestamp": datetime.now().isoformat(),
        "source_file": os.path.basename(input_json_path),
        "input_image": raw_data.get("input_path", ""),
        "model_settings": raw_data.get("model_settings", {}),
        "text_detection_params": raw_data.get("text_det_params", {}),
        "text_type": raw_data.get("text_type", ""),
        "rec_score_threshold": raw_data.get("text_rec_score_thresh", 0.0),
        "total_text_regions": len(raw_data.get("rec_texts", [])),
        
        # Main OCR results
        "rec_texts": raw_data.get("rec_texts", []),
        "rec_scores": raw_data.get("rec_scores", []),
    }
    
    # Create output filename
    input_dir = os.path.dirname(input_json_path)
    base_name = os.path.splitext(os.path.basename(input_json_path))[0]
    output_path = os.path.join(input_dir, f"{base_name}_simplified.json")
    
    # Save simplified data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(simplified_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Simplified OCR data extracted!")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Extracted {len(simplified_data['rec_texts'])} text regions")
    
    if simplified_data['rec_scores']:
        avg_confidence = sum(simplified_data['rec_scores']) / len(simplified_data['rec_scores'])
        print(f"🎯 Average confidence: {avg_confidence:.3f}")
    
    return output_path, simplified_data

if __name__ == "__main__":
    # Extract from the raw OCR JSON file
    input_file = "complete_raw_ocr_data/monitor_001_low_spo2_res.json"
    
    if os.path.exists(input_file):
        output_file, data = extract_simplified_ocr_data(input_file)
        
        # Display some key extracted data
        print("\n📝 Extracted Text Content:")
        for i, (text, score) in enumerate(zip(data['rec_texts'][:10], data['rec_scores'][:10])):
            print(f"  {i+1:2d}. '{text}' (confidence: {score:.3f})")
        
        if len(data['rec_texts']) > 10:
            print(f"  ... and {len(data['rec_texts']) - 10} more text regions")
            
    else:
        print(f"❌ Input file not found: {input_file}")
        print("📂 Available files in complete_raw_ocr_data:")
        try:
            for file in os.listdir("complete_raw_ocr_data"):
                if file.endswith('.json'):
                    print(f"   - {file}")
        except FileNotFoundError:
            print("   Directory not found")