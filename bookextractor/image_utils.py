from PIL import Image
import fitz
from typing import List

def crop_regions(img: Image.Image) -> List[Image.Image]:
    """
    Keep top, middle, bottom regions.
    """
    w, h = img.size
    top = img.crop((0, 0, w, h // 4))
    middle = img.crop((0, h // 4, w, 3 * h // 4))
    bottom = img.crop((0, 3 * h // 4, w, h))
    return [top, middle, bottom]

def crop_bbox(img: Image.Image, rect: fitz.Rect, page_rect: fitz.Rect) -> Image.Image:
    """
    Crop image using PyMuPDF Rect coordinates.
    """
    w, h = img.size
    pw, ph = page_rect.width, page_rect.height
    
    # Scale factors
    sx, sy = w / pw, h / ph
    
    left = rect.x0 * sx
    top = rect.y0 * sy
    right = rect.x1 * sx
    bottom = rect.y1 * sy
    
    return img.crop((left, top, right, bottom))

def combine_regions(standard_crops: List[Image.Image], keyword_crops: List[Image.Image]) -> List[Image.Image]:
    return standard_crops + keyword_crops
