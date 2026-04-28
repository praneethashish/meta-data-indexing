from typing import Any

import fitz
import piexif
from PIL import Image


def get_decimal_from_dms(dms, ref):
    """
    Convert DMS (Degrees, Minutes, Seconds) to decimal degrees.
    """
    degrees = dms[0][0] / dms[0][1]
    minutes = dms[1][0] / dms[1][1]
    seconds = dms[2][0] / dms[2][1]

    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ["S", "W"]:
        decimal = -decimal
    return decimal


def extract_image_metadata(img_path: str) -> dict[str, Any]:
    with Image.open(img_path) as img:
        width, height = img.size
        img_format = img.format
        dpi = img.info.get("dpi", (None, None))

        # Basic properties
        metadata = {
            "width": width,
            "height": height,
            "format": img_format,
            "dpi_horizontal": dpi[0],
            "dpi_vertical": dpi[1],
            "color_space": img.mode,
            "bit_depth": None,  # Pillow doesn't easily expose bit depth for all formats
            "exif_camera_make": None,
            "exif_camera_model": None,
            "exif_date_taken": None,
            "exif_gps_latitude": None,
            "exif_gps_longitude": None,
            "exif_lens": None,
        }

        # Bit depth estimation based on mode
        mode_to_bpp = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "YCbCr": 24, "I": 32, "F": 32}
        metadata["bit_depth"] = mode_to_bpp.get(img.mode)

        # EXIF data
        try:
            exif_dict = piexif.load(img.info.get("exif", b""))

            # 0th IFD
            if piexif.ImageIFD.Make in exif_dict["0th"]:
                metadata["exif_camera_make"] = exif_dict["0th"][piexif.ImageIFD.Make].decode("utf-8").strip("\x00")
            if piexif.ImageIFD.Model in exif_dict["0th"]:
                metadata["exif_camera_model"] = exif_dict["0th"][piexif.ImageIFD.Model].decode("utf-8").strip("\x00")
            if piexif.ImageIFD.DateTime in exif_dict["0th"]:
                metadata["exif_date_taken"] = exif_dict["0th"][piexif.ImageIFD.DateTime].decode("utf-8")

            # Exif IFD
            if piexif.ExifIFD.LensModel in exif_dict["Exif"]:
                metadata["exif_lens"] = exif_dict["Exif"][piexif.ExifIFD.LensModel].decode("utf-8").strip("\x00")

            # GPS IFD
            gps = exif_dict.get("GPS", {})
            if gps:
                if piexif.GPSIFD.GPSLatitude in gps and piexif.GPSIFD.GPSLatitudeRef in gps:
                    metadata["exif_gps_latitude"] = get_decimal_from_dms(
                        gps[piexif.GPSIFD.GPSLatitude], gps[piexif.GPSIFD.GPSLatitudeRef].decode("utf-8")
                    )
                if piexif.GPSIFD.GPSLongitude in gps and piexif.GPSIFD.GPSLongitudeRef in gps:
                    metadata["exif_gps_longitude"] = get_decimal_from_dms(
                        gps[piexif.GPSIFD.GPSLongitude], gps[piexif.GPSIFD.GPSLongitudeRef].decode("utf-8")
                    )
        except Exception:
            # If piexif fails or no exif, we still return what we found from Pillow
            pass

    return metadata


def crop_regions(img: Image.Image) -> list[Image.Image]:
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


def combine_regions(standard_crops: list[Image.Image], keyword_crops: list[Image.Image]) -> list[Image.Image]:
    return standard_crops + keyword_crops
