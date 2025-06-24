from paddleocr import PaddleOCR
import json
from datetime import datetime

def create_simple_json():
    """Create a simplified JSON with only rec_texts and rec_scores"""
    
    # Initialize OCR
    ocr = PaddleOCR(
        use_doc_orientation_classify=False, 
        use_doc_unwarping=False, 
        use_textline_orientation=False
    )
    
    # Process the image
    result = ocr.predict("medical_monitor.jpg")    # Extract just the text and scores
    simple_data = {
        "timestamp": datetime.now().isoformat(),
        "image_source": "medical_monitor.jpg",
        "rec_texts": [],
        "rec_scores": []
    }
    
    for res in result:
        # Access the data directly from the result object
        if 'rec_texts' in res:
            simple_data["rec_texts"] = res['rec_texts']
            simple_data["rec_scores"] = res['rec_scores'].tolist() if hasattr(res['rec_scores'], 'tolist') else res['rec_scores']
            break
    
    # Save simplified JSON
    with open("output/simple_ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(simple_data, f, indent=2, ensure_ascii=False)
    
    print("Simple OCR results saved to output/simple_ocr_results.json")
    print(f"Found {len(simple_data['rec_texts'])} text elements")
    
    # Display the results
    print("\nExtracted text with confidence scores:")
    for i, (text, score) in enumerate(zip(simple_data['rec_texts'], simple_data['rec_scores'])):
        print(f"{i:2d}: '{text}' (confidence: {score:.4f})")
    
    return simple_data

if __name__ == "__main__":
    create_simple_json()
