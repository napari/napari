"""Create test images

This is a throw-away file for creating a test image for octree rendering
development. If we keep test images in the product long term we'll
have a nicer way to generate them.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_text(image, text, nx=0.5, ny=0.5):

    font = ImageFont.truetype('Arial Black.ttf', size=72)
    (text_width, text_height) = font.getsize(text)
    x = nx * image.width - text_width / 2
    y = ny * image.height - text_height / 2

    color = 'rgb(255, 255, 255)'  # white

    draw = ImageDraw.Draw(image)
    draw.text((x, y), text, fill=color, font=font)
    draw.rectangle([0, 0, image.width, image.height], width=5)


def draw_text_tiled(image, text, nrows=1, ncols=1):

    print(f"Creating {nrows}x{ncols} text image: {text}")

    try:
        font = ImageFont.truetype('Arial Black.ttf', size=74)
    except OSError:
        font = ImageFont.load_default()
    (text_width, text_height) = font.getsize(text)

    color = 'rgb(255, 255, 255)'  # white
    draw = ImageDraw.Draw(image)

    for row in range(nrows + 1):
        for col in range(ncols + 1):
            x = (col / ncols) * image.width - text_width / 2
            y = (row / nrows) * image.height - text_height / 2

            draw.text((x, y), text, fill=color, font=font)
    draw.rectangle([0, 0, image.width, image.height], width=5)


def create_text_array(text, nx=0.5, ny=0.5, size=(1024, 1024)):
    text = str(text)
    image = Image.new('RGB', size)
    draw_text(image, text, nx, ny)
    return np.array(image)


def create_tiled_text_array(text, nrows, ncols, size=(1024, 1024)):
    text = str(text)
    image = Image.new('RGB', size)
    draw_text_tiled(image, text, nrows, ncols)
    return np.array(image)


def create_tiled_test_1(text, nrows, ncols, size=(1024, 1024)):
    text = str(text)
    image = Image.new('RGB', size)
    draw_text_tiled(image, text, nrows, ncols)
    return np.array(image)
