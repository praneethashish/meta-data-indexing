from typing import Any

from PIL import Image


def get_decimal_from_dms(value: tuple[tuple[int, int], tuple[int, int], tuple[int, int]], ref: str) -> float:
    """
    Helper function to convert the GPS coordinates stored in the EXIF to decimal degrees.
    """
    d = float(value[0][0]) / float(value[0][1])
    m = float(value[1][0]) / float(value[1][1])
    s = float(value[2][0]) / float(value[2][1])
    decimal = d + (m / 60.0) + (s / 3600.0)
    if ref in ["S", "W"]:
        return -decimal
    return decimal


def extract_image_metadata(image_path: str) -> dict[str, Any]:
    """
    Extracts metadata and EXIF data from an image file.
    """
    with Image.open(image_path) as img:
        width, height = img.size
        img_format = img.format
        dpi = img.info.get("dpi", (None, None))

        metadata = {
            "width": width,
            "height": height,
            "format": img_format or "UNKNOWN",
            "dpi_horizontal": float(dpi[0]) if dpi[0] is not None else None,
            "dpi_vertical": float(dpi[1]) if dpi[1] is not None else None,
            "color_space": img.mode,
            "bit_depth": None,
        }

        try:
            import piexif

            exif_dict = piexif.load(img.info.get("exif", b""))
            if exif_dict:
                zeroth = exif_dict.get("0th", {})
                make = zeroth.get(piexif.ImageIFD.Make, b"").decode("utf-8").strip("\x00") or None
                model = zeroth.get(piexif.ImageIFD.Model, b"").decode("utf-8").strip("\x00") or None
                metadata["exif_camera_make"] = make
                metadata["exif_camera_model"] = model

                exif = exif_dict.get("Exif", {})
                date_taken = exif.get(piexif.ExifIFD.DateTimeOriginal) or zeroth.get(piexif.ImageIFD.DateTime)
                if date_taken:
                    metadata["exif_date_taken"] = date_taken.decode("utf-8").strip("\x00")
                else:
                    metadata["exif_date_taken"] = None

                metadata["exif_lens"] = exif.get(piexif.ExifIFD.LensModel, b"").decode("utf-8").strip("\x00") or None

                gps = exif_dict.get("GPS", {})
                if gps:
                    lat = gps.get(piexif.GPSIFD.GPSLatitude)
                    lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef, b"N").decode("utf-8")
                    lon = gps.get(piexif.GPSIFD.GPSLongitude)
                    lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef, b"E").decode("utf-8")

                    if lat:
                        metadata["exif_gps_latitude"] = get_decimal_from_dms(lat, lat_ref)
                    if lon:
                        metadata["exif_gps_longitude"] = get_decimal_from_dms(lon, lon_ref)

        except Exception:  # nosec
            pass
    return metadata
