from vispy.color.colormap import Colormap as VispyColormap

from napari.utils.colormaps import Colormap


def napari_cmap_to_vispy(colormap: Colormap) -> VispyColormap:
    """Convert a napari colormap to its equivalent vispy colormap."""
    cmap_args = colormap.dict()
    cmap_args.pop('name')
    cmap_args['bad_color'] = cmap_args.pop('nan_color')
    return VispyColormap(**cmap_args)
