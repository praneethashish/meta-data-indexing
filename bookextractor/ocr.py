import easyocr
import numpy as np
from PIL import Image
from typing import List
from .validation import extract_isbn_candidates

class OCRScanner:
    def __init__(self):
        # EasyOCR has compatibility limits (e.g., Telugu only with English).
        # We use two readers to cover both Hindi and Telugu.
        self.readers = {
            'hi': easyocr.Reader(['hi', 'en']),
            'te': easyocr.Reader(['te', 'en'])
        }

    def scan_image(self, img: Image.Image) -> str:
        img_np = np.array(img)
        texts = []
        for lang, reader in self.readers.items():
            results = reader.readtext(img_np)
            texts.append(" ".join([text for (bbox, text, prob) in results]))
        
        # Merge unique words/phrases from both scans
        return " ".join(list(dict.fromkeys(" ".join(texts).split())))

    def find_isbns(self, crops: List[Image.Image]) -> List[str]:
        all_isbns = []
        for crop in crops:
            text = self.scan_image(crop)
            isbns = extract_isbn_candidates(text)
            all_isbns.extend(isbns)
        return list(set(all_isbns))
