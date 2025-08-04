"""Example script for starting napari with a custom colormap

This is an example script that could be provided in napari setting
to be executed on napari application startup.

This script adds a custom colormap named 'cyan_t' to the napari
"""

from napari.utils.colormaps import Colormap, ensure_colormap

custom_cyan_colormap = Colormap(colors=ensure_colormap('cyan').colors, name='cyan_t', low_color=[0, 0, 0, 0])
ensure_colormap(custom_cyan_colormap)
