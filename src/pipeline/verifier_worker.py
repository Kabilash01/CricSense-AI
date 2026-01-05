# Verifier worker implementation
import easyocr
import cv2

# Load image
img = cv2.imread(r"C:\cricket-ai\jersey_crops\SID_9\DEBUG_f147_r1.jpg")

# Initialize reader (English)
reader = easyocr.Reader(['en'], gpu=False)

# OCR
results = reader.readtext(img)

# Print detected text
for bbox, text, confidence in results:
    print(f"{text}  | Confidence: {confidence:.2f}")
