import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


class SmolVLMJerseyReader:
    """
    Jersey number reader using PUBLIC SmolVLM-256M-Instruct.
    Requires <image> token in prompt.
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        self.model.eval()

    def read_jersey_number(self, image_path):
        """
        Args:
            image_path (Path or str)

        Returns:
            (number: str | None, raw_text: str)
        """

        image = Image.open(image_path).convert("RGB")

        # ⚠️ <image> token is MANDATORY for SmolVLM
        prompt = (
            "<image>\n"
            "Look at the player's jersey. "
            "If a jersey number is visible, answer with ONLY the number. "
            "If no number is visible, answer None."
        )

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )

        # Move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False
            )

        decoded = self.processor.batch_decode(
            outputs, skip_special_tokens=True
        )[0].strip()

        # Extract digits safely
        match = re.search(r"\d+", decoded)
        if match:
            return match.group(), decoded

        return None, decoded
