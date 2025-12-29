import re
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


class SmolVLMJerseyReader:
    """
    Jersey number reader using SmolVLM.
    Supports both file paths and NumPy image arrays.
    Optimized for low latency.
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

    def _to_pil(self, img):
        if isinstance(img, Image.Image):
            return img.convert("RGB")

        if isinstance(img, np.ndarray):
            # OpenCV BGR → RGB
            return Image.fromarray(img[:, :, ::-1]).convert("RGB")

        return Image.open(img).convert("RGB")

    def read_jersey_number(self, image_input):
        image = self._to_pil(image_input)

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
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4,   # 🔥 reduced for speed
                do_sample=False
            )

        decoded = self.processor.batch_decode(
            outputs, skip_special_tokens=True
        )[0].strip()

        match = re.search(r"\d+", decoded)
        if match:
            return match.group(), decoded

        return None, decoded
