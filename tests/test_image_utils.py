import piexif
from PIL import Image

from bookextractor.image_utils import extract_image_metadata, get_decimal_from_dms


def test_get_decimal_from_dms():
    # 34 deg 3 min 11 sec N
    dms = ((34, 1), (3, 1), (11, 1))
    assert abs(get_decimal_from_dms(dms, "N") - 34.0530555) < 0.0001
    # 118 deg 14 min 45 sec W
    dms = ((118, 1), (14, 1), (45, 1))
    assert abs(get_decimal_from_dms(dms, "W") - (-118.2458333)) < 0.0001


def test_extract_image_metadata(tmp_path):
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
