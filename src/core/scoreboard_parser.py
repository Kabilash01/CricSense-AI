import re

class ScoreboardParser:
    """
    Parses raw OCR text lines into structured scoreboard fields.
    NO OCR here.
    """

    def parse(self, ocr_lines):
        """
        ocr_lines: list[str] from OCRWorker.get_latest()
        """

        text = " ".join(ocr_lines).upper()

        state = {
            "team_score": None,
            "overs": None,
            "striker": None,
            "non_striker": None,
            "bowler": None
        }

        # --- Team score (e.g. 123-4 or 123/4) ---
        m = re.search(r"(\d{1,3})\s*[-/]\s*(\d)", text)
        if m:
            state["team_score"] = f"{m.group(1)}-{m.group(2)}"

        # --- Overs (e.g. 12.3) ---
        m = re.search(r"\b(\d{1,2}\.\d)\b", text)
        if m:
            state["overs"] = m.group(1)

        # --- Names (best effort, Phase-1) ---
        tokens = text.split()

        # heuristic: last word often bowler
        if len(tokens) >= 1:
            state["bowler"] = tokens[-1]

        # striker / non-striker will be stabilized later
        return state
