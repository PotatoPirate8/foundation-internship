from paddleocr import PaddleOCR
import json

ocr = PaddleOCR(
    use_doc_orientation_classify=False, 
    use_doc_unwarping=False, 
    use_textline_orientation=False) # text detection + text recognition

result = ocr.predict("medical_monitor.jpg")

# Save simplified JSON with just text and scores
for res in result:
    simple_data = {
        "rec_texts": res['rec_texts'],
        "rec_scores": res['rec_scores'].tolist() if hasattr(res['rec_scores'], 'tolist') else res['rec_scores']
    }
    
    with open("output/simple_ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(simple_data, f, indent=2, ensure_ascii=False)
    
    # Also save the full result
    res.save_to_img("output")
    res.save_to_json("output")
    
    print(f"Saved simple JSON with {len(simple_data['rec_texts'])} text elements")
    break