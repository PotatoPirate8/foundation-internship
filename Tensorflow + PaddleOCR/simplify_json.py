import json
import os
from datetime import datetime


def extract_simplified_ocr_data(input_json_path):
    """Extract rec_texts, rec_scores and important metadata from raw OCR JSON"""

    # Read the raw OCR data
    with open(input_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    box_midpoints = []
    poly_midpoints = []
    # Calculate midpoints for potential distance grouping
    if 'rec_boxes' in raw_data and 'rec_polys' in raw_data:
        boxes = raw_data['rec_boxes']
        texts = raw_data['rec_texts']

        for i, (box, text) in enumerate(zip(boxes, texts)):
            x1, y1, x2, y2 = box
            box_midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
            box_midpoints.append({"x": box_midpoint[0], "y": box_midpoint[1]})
            # for i, (poly, text) in enumerate(zip(polys, texts)): # Find centroid
            # Hmm they use the same formula, this makes things easier (Edit: I lied)

            if 'rec_polys' in raw_data and i < len(raw_data['rec_polys']):
                poly = raw_data['rec_polys'][i]
                poly_x = [p[0] for p in poly]
                poly_y = [p[1] for p in poly]
                poly_midpoint = (sum(poly_x) / len(poly), sum(poly_y) / len(poly))

                # if poly_midpoint != box_midpoint:
                poly_midpoints.append({
                    "x": poly_midpoint[0],
                    "y": poly_midpoint[1]
                })

    # Extract important information
    simplified_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "source_file": os.path.basename(input_json_path),
            "input_image": raw_data.get("input_path", ""),
            "text_type": raw_data.get("text_type", ""),
            "rec_score_threshold": raw_data.get("text_rec_score_thresh", 0.0),
            "total_text_regions": len(raw_data.get("rec_texts", []))
        },
        "model_info": {
            "model_settings": raw_data.get("model_settings", {}),
            "detection_params": raw_data.get("text_det_params", {})
        },
        "ocr_results": [{
            "text": text,
            "confidence": score,
            "box_midpoint": box_midpoints[i] if i < len(box_midpoints) else None,
            "poly_midpoint": poly_midpoints[i] if i < len(poly_midpoints) else None
        } for i, (text, score) in enumerate(zip(
            raw_data.get("rec_texts", []),
            raw_data.get("rec_scores", [])
        ))]
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
    print(f"📊 Extracted {len(simplified_data['ocr_results'])} text regions")

    if simplified_data['ocr_results'] and simplified_data['ocr_results'][0]['confidence'] is not None:
        avg_confidence = sum(r['confidence'] for r in simplified_data['ocr_results']) / len(
            simplified_data['ocr_results'])
        print(f"🎯 Average confidence: {avg_confidence:.3f}")

    return output_path, simplified_data


if __name__ == "__main__":
    # Extract from the raw OCR JSON file
    input_file = "complete_raw_ocr_data/medical_monitor_res.json"

    if os.path.exists(input_file):
        output_file, data = extract_simplified_ocr_data(input_file)

        # Display some key extracted data
        print("\n📝 Extracted Text Content:")
        for i, result in enumerate(data['ocr_results'][:10]):  # First 10 results
            print(f"  {i + 1:2d}. '{result['text']}' (confidence: {result['confidence']})")

        if len(data['ocr_results']) > 10:
            print(f"  ... and {len(data['ocr_results']) - 10} more text regions")

    else:
        print(f"❌ Input file not found: {input_file}")
        print("📂 Available files in complete_raw_ocr_data:")
        try:
            for file in os.listdir("complete_raw_ocr_data"):
                if file.endswith('.json'):
                    print(f"   - {file}")
        except FileNotFoundError:
            print("   Directory not found")
