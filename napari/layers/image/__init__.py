from ._async_image import AsyncImage
from .image import Image

from ...utils.chunk import async_config

# Use AsyncImage only if configured to use async.
if not async_config.synchronous:
    Image = AsyncImage
