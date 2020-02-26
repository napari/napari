from ..layers import Image
from .image_controller import ImageController

layer_to_controller = {
    Image: ImageController,
}
