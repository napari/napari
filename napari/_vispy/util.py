from .vispy_base_layer import VispyBaseLayer


def create_vispy_visual(layer):
    """
    Create vispy visual for a layer based on its layer type.

    Parameters
    ----------
        layer : napari.layers._base_layer.Layer
            Layer that needs its propetry widget created.

    Returns
    ----------
        visual : vispy.scene.visuals.VisualNode
            Vispy visual node
    """
    return VispyBaseLayer(layer)
