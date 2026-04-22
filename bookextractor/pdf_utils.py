import fitz  # PyMuPDF
from typing import List

def extract_keyword_bboxes(page: fitz.Page, keywords: List[str] = None) -> List[fitz.Rect]:
    """
    Find lines containing keywords and return padded bounding boxes.
    """
    if keywords is None:
        keywords = ["isbn", "publisher", "published", "edition", "year", "date"]
    
    bboxes = []
    text_instances = []
    
    for kw in keywords:
        # Search for keyword (case-insensitive search is default in newer PyMuPDF or use search_for)
        instances = page.search_for(kw)
        text_instances.extend(instances)
    
    for rect in text_instances:
        # Expand bbox slightly (padding: 50 pixels horizontally, 20 vertically)
        padded_rect = fitz.Rect(
            max(0, rect.x0 - 50),
            max(0, rect.y0 - 20),
            min(page.rect.width, rect.x1 + 400), # ISBNs usually follow the keyword
            min(page.rect.height, rect.y1 + 20)
        )
        bboxes.append(padded_rect)
        
    return bboxes

def get_page_image(page: fitz.Page, dpi: int = 300):
    pix = page.get_pixmap(dpi=dpi)
    return pix
