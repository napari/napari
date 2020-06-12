import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_text_image(text: str, rgb: bool):
    """Return blank image with given text centered on it.

    TODO_ASYNC: Use vispy text not PIL.

    Parameters
    ----------
    text : str
        The text to put in the center of the image.
    rgb : bool
        Is the image RGB format? Otherwise grayscale.
    """
    size = (1024, 1024)

    if rgb:
        image = Image.new('RGB', size)
    else:
        image = Image.new('L', size)

    font = ImageFont.truetype('Arial Black.ttf', size=72)
    (width, height) = font.getsize(text)
    x = (image.width / 2) - width / 2
    y = (image.height / 2) - height / 2

    color = 'rgb(255, 255, 255)'  # white color
    draw = ImageDraw.Draw(image)
    draw.text((x, y), text, fill=color, font=font)

    return np.array(image)
