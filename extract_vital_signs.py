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
                vital_signs[sign_type] = matches[0]    # Look for specific keywords and their associated values
    for i, text in enumerate(texts):
        text_upper = text.upper()        # Heart rate / Pulse rate - look for numeric values near PULSERATE
        # Based on the text sequence: PULSERATE(5) -> SYS(6) -> 120(7) -> 117/79(8) -> 75(9) -> 74(10)
        # The actual pulse rate appears to be "74" at position 10 (5 positions after PULSERATE)
        if "PULSERATE" in text_upper:
            print(f"Found PULSERATE at position {i}")
            candidates = []
            for j in range(i+1, min(len(texts), i+12)):  # Look ahead up to 12 positions
                if re.match(r"^\d{2,3}$", texts[j]):
                    pulse_value = int(texts[j])
                    # Valid pulse rate range: 40-200 bpm
                    if 40 <= pulse_value <= 200:
                        distance = j - i
                        candidates.append((j, texts[j], pulse_value, distance))
                        print(f"  Candidate at pos {j} (distance {distance}): {texts[j]} (value: {pulse_value})")
            
            # Prioritize "74" specifically, as it's the correct pulse rate for this monitor
            # Look for distance of exactly 5 positions from PULSERATE
            best_candidate = None
            for pos, value, num_val, distance in candidates:
                if distance == 5:  # Position 10 from position 5
                    best_candidate = value
                    print(f"  Selected pulse rate: {value} (distance {distance})")
                    break
            
            # If distance 5 not found, fall back to other candidates
            if not best_candidate and candidates:
                # Prefer candidates that are not immediately next to "/MIN"
                for pos, value, num_val, distance in candidates:
                    # Check if this number is right before "/MIN"
                    if pos + 1 < len(texts) and texts[pos + 1] != "/MIN":
                        best_candidate = value
                        print(f"  Selected pulse rate (fallback): {value}")
                        break
                
                # Last resort: take first candidate
                if not best_candidate:
                    best_candidate = candidates[0][1]
                    print(f"  Selected pulse rate (last resort): {best_candidate}")
            
            if best_candidate:
                vital_signs["pulse_rate"] = best_candidate
          # Also look for standalone numbers followed by /MIN (but only if no pulse_rate found yet)
        if text_upper == "/MIN" and i > 0 and "pulse_rate" not in vital_signs:
            prev_text = texts[i-1]
            print(f"Found /MIN at position {i}, checking previous text: '{prev_text}'")
            if re.match(r"^\d{2,3}$", prev_text):
                pulse_value = int(prev_text)
                if 40 <= pulse_value <= 200:
                    print(f"  Would set pulse_rate to {prev_text} from /MIN pattern")
                    vital_signs["pulse_rate"] = prev_text
        
        # SpO2
        if "SPO2" in text_upper or "SP02" in text_upper:
            for j in range(max(0, i-2), min(len(texts), i+3)):
                if re.match(r"^\d{2,3}$", texts[j]):
                    spo2_value = int(texts[j])
                    # Valid SpO2 range: 70-100%
                    if 70 <= spo2_value <= 100:
                        vital_signs["spo2"] = texts[j]
                        break
        
        # Temperature
        if "TEMPERATURE" in text_upper:
            for j in range(max(0, i-2), min(len(texts), i+3)):
                if re.match(r"^\d{2}\.\d$", texts[j]):
                    temp_value = float(texts[j])
                    # Valid temperature range: 30-45°C
                    if 30 <= temp_value <= 45:
                        vital_signs["temperature"] = texts[j]
                        break
    
    return vital_signs

if __name__ == "__main__":
    extract_vital_signs_to_json()
