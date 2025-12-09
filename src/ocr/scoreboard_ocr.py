# Scoreboard OCR implementation
# src/ocr/scoreboard_ocr.py
import pytesseract
import cv2
import re

# Ensure tesseract is in PATH on Windows, or set pytesseract.pytesseract.tesseract_cmd

def preprocess_scoreboard(crop):
    # crop: BGR scoreboard region
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # optional resize and bilateral filter
    h,w = gray.shape[:2]
    if max(h,w) < 200:
        gray = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def parse_score_text(text):
    # tries to find patterns like "123/4" and "17.2"
    runs = None; wickets = None; overs = None
    m = re.search(r'(\d{1,3})\s*\/\s*(\d{1,2})', text)
    if m:
        runs, wickets = int(m.group(1)), int(m.group(2))
    m2 = re.search(r'(\d{1,2})[.:](\d)', text)
    if m2:
        overs = float(f"{m2.group(1)}.{m2.group(2)}")
    return {"raw": text, "runs": runs, "wickets": wickets, "overs": overs}

def read_scoreboard(crop):
    img = preprocess_scoreboard(crop)
    # use pytesseract (fast) or easyocr
    config = "--psm 6 -c tessedit_char_whitelist=0123456789/.:"
    text = pytesseract.image_to_string(img, config=config)
    parsed = parse_score_text(text)
    return parsed
