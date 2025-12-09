# Jersey number OCR implementation
# src/ocr/jersey_ocr.py
import easyocr
import cv2
import numpy as np
from collections import deque, Counter

reader = easyocr.Reader(['en'], gpu=True)  # set gpu=False if no GPU

def preprocess_for_jersey(crop):
    # crop: BGR image of torso region; return grayscale resized for OCR
    h,w = crop.shape[:2]
    # focus on central torso area (avoid arms)
    y1 = int(h*0.25); y2 = int(h*0.75)
    x1 = int(w*0.2); x2 = int(w*0.8)
    crop_torso = crop[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop_torso, cv2.COLOR_BGR2GRAY)
    # adaptive threshold for contrast
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,11,2)
    # resize to improve small digits
    th = cv2.resize(th, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return th

def read_jersey_number(crop):
    """
    crop: BGR bbox centered on torso
    returns: list of candidate strings with confidences
    """
    img = preprocess_for_jersey(crop)
    # EasyOCR expects RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    results = reader.readtext(img_rgb, detail=1, paragraph=False)
    # results: list of (bbox, text, confidence)
    candidates = []
    for bbox, text, conf in results:
        # filter non-digit characters
        digits = ''.join([c for c in text if c.isdigit()])
        if digits:
            candidates.append((digits, float(conf)))
    # sort by confidence
    candidates.sort(key=lambda x: -x[1])
    return candidates  # maybe empty
