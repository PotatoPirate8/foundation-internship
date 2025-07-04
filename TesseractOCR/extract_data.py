import re

import cv2
import pytesseract
from scipy.spatial import distance

COMMON_KEYWORDS = [
    "nibp", "temperature", "pulse rate", "rn", "sp02"
]

def extract_text_with_boxes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # Compute threshold

    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)  # Main data extraction

    boxes = []

    for i in range(len(data["text"])):  # Process data
        if int(data["conf"][i] > 60) and data["text"][i].strip():
            x, y, w, h, = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            text = data["text"][i].strip()
            boxes.append((x, y, w, h, text))

    return boxes


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

#
# boxes = [
#     (100, 50, 40, 20, "HR"),    # Label
#     (150, 55, 30, 20, "72"),    # Value (paired with "HR")
#     (200, 50, 50, 20, "SpO2"),  # Label
#     (260, 55, 30, 20, "98"),    # Value (paired with "SpO2")
#     (100, 100, 60, 20, "BP"),   # Label
#     (170, 105, 50, 20, "120/80") # Value (paired with "BP")
# ]


boxes = extract_text_with_boxes("medical_monitor.jpg")
print(boxes)
pairs = pair_label_with_value(boxes)
