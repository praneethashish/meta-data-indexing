from unittest.mock import patch

import piexif
from PIL import Image

from bookextractor.image_utils import (
    combine_regions,
    crop_bbox,
    crop_regions,
    extract_image_metadata,
    get_decimal_from_dms,
)


def test_get_decimal_from_dms():
    # 34 deg 3 min 11 sec N
    dms = ((34, 1), (3, 1), (11, 1))
    assert abs(get_decimal_from_dms(dms, "N") - 34.0530555) < 0.0001
    # 118 deg 14 min 45 sec W
    dms = ((118, 1), (14, 1), (45, 1))
    assert abs(get_decimal_from_dms(dms, "W") - (-118.2458333)) < 0.0001


def test_crop_regions():
    img = Image.new("RGB", (100, 100))
    regions = crop_regions(img)
    assert len(regions) == 3
    assert regions[0].size == (100, 25)
    assert regions[1].size == (100, 50)
    assert regions[2].size == (100, 25)


def test_crop_bbox():
    from unittest.mock import MagicMock

    img = Image.new("RGB", (100, 100))
    rect = MagicMock()
    rect.x0, rect.y0, rect.x1, rect.y1 = 0, 0, 50, 50
    page_rect = MagicMock()
    page_rect.width, page_rect.height = 100, 100

    cropped = crop_bbox(img, rect, page_rect)
    assert cropped.size == (50, 50)


def test_combine_regions():
    a = [Image.new("RGB", (1, 1))]
    b = [Image.new("RGB", (1, 1))]
    combined = combine_regions(a, b)
    assert len(combined) == 2


def test_extract_image_metadata_no_date(tmp_path):
    # Test line 90 (no date_taken)
    test_img_path = str(tmp_path / "test_no_date.jpg")
    img = Image.new("RGB", (10, 10))

    # EXIF with no date
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}}
    exif_bytes = piexif.dump(exif_dict)
    img.save(test_img_path, exif=exif_bytes)

    metadata = extract_image_metadata(test_img_path)
    assert metadata["exif_date_taken"] is None


def test_extract_image_metadata_no_dpi_no_exif(tmp_path):
    # Test line 90 (no dpi) and lines 107-108 (corrupt EXIF)
    test_img_path = str(tmp_path / "test_no_info.jpg")
    img = Image.new("RGB", (10, 10))
    # DPI is missing by default
    img.save(test_img_path)

    metadata = extract_image_metadata(test_img_path)
    assert metadata["dpi_horizontal"] is None

    # Trigger exception in line 107-108
    with patch("piexif.load", side_effect=Exception("Corrupt EXIF")):
        metadata = extract_image_metadata(test_img_path)
        # Should return metadata despite EXIF error
        assert metadata["width"] == 10
    test_img_path = str(tmp_path / "test_image.jpg")

    # Create a simple RGB image
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))

    # Prepare EXIF data
    zeroth_ifd = {
        piexif.ImageIFD.Make: "TestMake",
        piexif.ImageIFD.Model: "TestModel",
        piexif.ImageIFD.DateTime: "2023:10:27 10:00:00",
    }
    exif_ifd = {
        piexif.ExifIFD.LensModel: "TestLens",
    }
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: "N",
        piexif.GPSIFD.GPSLatitude: ((34, 1), (3, 1), (11, 1)),
        piexif.GPSIFD.GPSLongitudeRef: "W",
        piexif.GPSIFD.GPSLongitude: ((118, 1), (14, 1), (45, 1)),
    }

    exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd}
    exif_bytes = piexif.dump(exif_dict)

    img.save(test_img_path, exif=exif_bytes)

    metadata = extract_image_metadata(test_img_path)

    assert metadata["width"] == 100
    assert metadata["height"] == 100
    assert metadata["exif_camera_make"] == "TestMake"
    assert metadata["exif_camera_model"] == "TestModel"
    assert metadata["exif_lens"] == "TestLens"
    assert metadata["exif_date_taken"] == "2023:10:27 10:00:00"
    assert abs(metadata["exif_gps_latitude"] - 34.0530555) < 0.0001
    assert abs(metadata["exif_gps_longitude"] - (-118.2458333)) < 0.0001
