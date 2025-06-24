from paddleocr import PaddleOCR
import json
import re
from datetime import datetime

def extract_vital_signs_to_json():
    """Extract vital signs from medical monitor and save to clean JSON format"""
    
    # Initialize OCR
    ocr = PaddleOCR(
        use_doc_orientation_classify=True, 
        use_doc_unwarping=True, 
        use_textline_orientation=False
    )
    
    # Process the image
    result = ocr.predict("medical_monitor.jpg")    # Extract text from OCR results
    extracted_texts = []
    for res in result:
        # Access rec_texts directly as a key
        if 'rec_texts' in res:
            extracted_texts.extend(res['rec_texts'])
        else:
            print(f"Warning: 'rec_texts' not found in result")
            print(f"Available keys: {list(res.keys())}")
            break
    
    # Parse vital signs from extracted text
    vital_signs = parse_vital_signs(extracted_texts)
    
    # Create structured output
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "image_source": "medical_monitor.jpg",
        "vital_signs": vital_signs,
        "raw_extracted_text": extracted_texts,
        "total_text_elements": len(extracted_texts)
    }
    
    # Save to JSON file
    with open("output/vital_signs_extracted.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("Vital signs extracted and saved to output/vital_signs_extracted.json")
    print(f"Found {len(extracted_texts)} text elements")
    print("\nExtracted vital signs:")
    for key, value in vital_signs.items():
        print(f"  {key}: {value}")
    
    return output_data

def parse_vital_signs(texts):
    """Parse vital signs from extracted text"""
    vital_signs = {}
      # Join all texts for easier pattern matching
    full_text = " ".join(texts)
    
    # Common vital sign patterns
    patterns = {
        "blood_pressure": r"(\d{2,3})/(\d{2,3})(?!/\d{4})",  # e.g., 117/79 but not dates like 21/10/2010
        "pulse_rate": r"(\d{2,3})\s*/MIN",  # e.g., 75 /MIN
        "spo2": r"(\d{2,3})\s*%",  # e.g., 100%
        "temperature_celsius": r"(\d{2}\.\d)\s*°?[C℃]",  # e.g., 38.3°C
        "temperature_fahrenheit": r"(\d{2,3}\.\d)\s*°?F",  # e.g., 97.1°F
        "patient_id": r"RN:(\d+)",  # e.g., RN:33398968
        "timestamp": r"(\d{2}:\d{2})",  # e.g., 15:13
        "date": r"(\d{1,2}/\d{1,2}/\d{4})"  # e.g., 21/10/2010
    }
    
    for sign_type, pattern in patterns.items():
        matches = re.findall(pattern, full_text)
        if matches:
            if sign_type == "blood_pressure":
                # Filter out dates and find actual blood pressure readings
                valid_bp = []
                for match in matches:
                    systolic, diastolic = int(match[0]), int(match[1])
                    # Blood pressure ranges: systolic 70-200, diastolic 40-120
                    if 70 <= systolic <= 200 and 40 <= diastolic <= 120:
                        valid_bp.append(match)
                
                if valid_bp:
                    vital_signs["systolic_bp"] = valid_bp[0][0]
                    vital_signs["diastolic_bp"] = valid_bp[0][1]
                    vital_signs["blood_pressure"] = f"{valid_bp[0][0]}/{valid_bp[0][1]}"
            elif sign_type in ["pulse_rate", "spo2", "temperature_celsius", "temperature_fahrenheit"]:
                vital_signs[sign_type] = matches[0]
            else:
                vital_signs[sign_type] = matches[0]
    
    # Look for specific keywords and their associated values
    for i, text in enumerate(texts):
        text_upper = text.upper()
        
        # Heart rate / Pulse rate
        if "PULSERATE" in text_upper and i + 1 < len(texts):
            try:
                vital_signs["pulse_rate"] = texts[i + 1]
            except:
                pass
        
        # SpO2
        if "SPO2" in text_upper or "SP02" in text_upper:
            for j in range(max(0, i-2), min(len(texts), i+3)):
                if re.match(r"^\d{2,3}$", texts[j]):
                    vital_signs["spo2"] = texts[j]
                    break
        
        # Temperature
        if "TEMPERATURE" in text_upper and i + 1 < len(texts):
            try:
                vital_signs["temperature"] = texts[i + 1]
            except:
                pass
    
    return vital_signs

if __name__ == "__main__":
    extract_vital_signs_to_json()
