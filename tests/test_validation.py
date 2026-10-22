import io

import numpy as np
from PIL import Image, ImageDraw

from src.validation import INVALID_UPLOAD_MESSAGE, validate_image, validate_upload


def handwritten_digit(background=255, foreground=0, xy=(44, 12)):
    image = Image.new("L", (100, 100), background)
    ImageDraw.Draw(image).line((xy[0], xy[1], xy[0], xy[1] + 76), fill=foreground, width=10)
    return image


def test_valid_mnist_like_png_and_jpeg_uploads():
    for fmt in ("PNG", "JPEG"):
        stream = io.BytesIO()
        handwritten_digit().save(stream, format=fmt)
        stream.seek(0)
        result = validate_upload(stream)
        assert result.valid


def test_transparent_background_with_opaque_digit_is_accepted():
    image = Image.new("RGBA", (100, 100), (255, 255, 255, 0))
    ImageDraw.Draw(image).line((45, 15, 45, 85), fill=(0, 0, 0, 255), width=10)
    assert validate_image(image).valid


def test_unsupported_extension_and_oversized_upload_are_rejected():
    stream = io.BytesIO()
    handwritten_digit().save(stream, format="PNG")
    stream.name = "digit.gif"
    assert not validate_upload(stream).valid


def test_blank_gray_transparent_corrupt_and_unsupported_are_rejected():
    assert not validate_image(Image.new("L", (80, 80), 255)).valid
    assert not validate_image(Image.new("L", (80, 80), 128)).valid
    assert not validate_image(Image.new("RGBA", (80, 80), (0, 0, 0, 0))).valid
    assert validate_upload(io.BytesIO(b"not an image")).message == INVALID_UPLOAD_MESSAGE
    assert not validate_image(np.zeros((80, 80, 2), dtype=np.uint8)).valid


def test_leaf_and_object_like_photos_are_rejected():
    leaf = np.zeros((100, 100, 3), dtype=np.uint8)
    leaf[:] = (30, 150, 50)
    yy, xx = np.indices((100, 100))
    leaf[((xx - 50) / 35) ** 2 + ((yy - 50) / 20) ** 2 < 1] = (20, 90, 30)
    assert not validate_image(Image.fromarray(leaf)).valid

    object_like = Image.new("L", (100, 100), 230)
    ImageDraw.Draw(object_like).rectangle((20, 20, 80, 80), fill=100)
    assert not validate_image(object_like).valid


def test_dark_background_and_off_center_digit_remain_valid():
    assert validate_image(handwritten_digit(20, 240)).valid
    assert validate_image(handwritten_digit(xy=(72, 12))).valid


def test_two_foreground_objects_are_rejected():
    image = Image.new("L", (100, 100), 255)
    draw = ImageDraw.Draw(image)
    draw.ellipse((12, 30, 40, 70), fill=0)
    draw.ellipse((60, 30, 88, 70), fill=0)
    assert not validate_image(image).valid
