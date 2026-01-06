import re

class ScoreboardNormalizer:

    @staticmethod
    def normalize_overs(text):
        if not text:
            return None
        # Fix common OCR mistakes
        text = text.replace(" ", "")
        text = text.replace("O", "0")
        text = text.replace("l", "1")

        # Match formats like 0.1/20 or 18.3
        match = re.search(r"\d+\.\d+(/\d+)?", text)
        return match.group(0) if match else None

    @staticmethod
    def normalize_score(text):
        if not text:
            return None
        match = re.search(r"\d+-\d+", text)
        return match.group(0) if match else None

    @staticmethod
    def normalize_player(text):
        if not text:
            return None
        # Remove junk characters
        text = re.sub(r"[^A-Za-z0-9() ]", "", text)
        return text.strip() if len(text.strip()) > 2 else None
