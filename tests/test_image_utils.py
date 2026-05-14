import pytest

try:
    import piexif
    from PIL import Image
except ImportError:
    piexif = None  # type: ignore
    Image = None  # type: ignore

from unittest.mock import patch

from bookextractor.image_utils import (
    extract_image_metadata,
    get_decimal_from_dms,
)


def test_get_decimal_from_dms():
    dms = ((34, 1), (3, 1), (11, 1))
    assert abs(get_decimal_from_dms(dms, "N") - 34.0530555) < 0.0001
    dms = ((118, 1), (14, 1), (45, 1))
    assert abs(get_decimal_from_dms(dms, "W") - (-118.2458333)) < 0.0001


@pytest.mark.skipif(Image is None or piexif is None, reason="PIL or piexif not installed")
def test_extract_image_metadata_no_date(tmp_path):
    test_img_path = str(tmp_path / "test_no_date.jpg")
    img = Image.new("RGB", (10, 10))

    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}}
    exif_bytes = piexif.dump(exif_dict)
    img.save(test_img_path, exif=exif_bytes)

    metadata = extract_image_metadata(test_img_path)
    assert metadata["exif_date_taken"] is None


@pytest.mark.skipif(Image is None, reason="PIL not installed")
def test_extract_image_metadata_no_dpi_no_exif(tmp_path):
    test_img_path = str(tmp_path / "test_no_info.jpg")
    img = Image.new("RGB", (10, 10))
    img.save(test_img_path)

    metadata = extract_image_metadata(test_img_path)
    assert metadata["dpi_horizontal"] is None

    with patch("piexif.load", side_effect=Exception("Corrupt EXIF")):
        metadata = extract_image_metadata(test_img_path)
        assert metadata["width"] == 10


@pytest.mark.skipif(Image is None or piexif is None, reason="PIL or piexif not installed")
def test_extract_image_metadata_full(tmp_path):
    test_img_path = str(tmp_path / "test_image.jpg")

    img = Image.new("RGB", (100, 100), color=(73, 109, 137))

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
