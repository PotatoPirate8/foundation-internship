import cv2
import easyocr
reader = easyocr.Reader(["en"], gpu=True)

def extract_data(image_path):
    """Extract data from image"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    results = reader.readtext(gray, paragraph=False)

    boxes = []
    for (bbox, text, confidence) in results:
        if confidence > 0.6:
            x, y = map(int, bbox[0])
            w, h = int(bbox[2][0] - x), int(bbox[2][1] - y)
            boxes.append((x, y, w, h, text.strip()))

    return boxes

COMMON_KEYWORDS = [
    "nibp", "temperature", "pulse rate", "rn", "spo2"
]

def pair_label_with_value(boxes):
    """Pair value with label based on their proximity"""

    labels = []
    values = []

    for (x, y, w, h, text) in boxes:  # Separate into value and label
        text = text.strip()

        if text.replace(".", "").replace("/", "").isdigit(): # If strictly numerals
            values.append((x,y,w,h, text))
        elif any(c.isalpha() for c in text):
            labels.append((x,y,w,h, text))

    pairs = {}
    #
    print("Labels" , labels)
    print("Values" , values)

    for (lx, ly, lw, lh, label) in labels:
        for keyword in COMMON_KEYWORDS:
            if keyword in label.lower().strip():
                min_dist = float("inf")
                closest_value = None
                closest_pos = None  # For debugging

                label_center = (lx + lw / 2, ly + lh / 2)

                for (vx, vy, vw, vh, value) in values:

                    value_center = (vx + vw / 2, vy + vh / 2)
                    dist = ((label_center[0] - value_center[0])**2 +
                            (label_center[1] - value_center[1])**2)**0.5

                    if dist < min_dist:
                        min_dist = dist
                        closest_value = value
                        closest_pos = (vx, vy)


                if closest_value:
                    print(f"Paired: '{label}' ({label_center}) -> '{closest_value}' ({closest_pos}) | Distance: {min_dist:.1f}")
                    pairs[label] = closest_value

    return pairs


boxes = extract_data("medical_monitor.jpg")
pairs = pair_label_with_value(boxes)


print(pairs)
