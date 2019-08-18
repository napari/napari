from .vispy_base_layer import VispyBaseLayer
from .vispy_image_layer import VispyImageLayer
from .vispy_labels_layer import VispyLabelsLayer
from .vispy_vectors_layer import VispyVectorsLayer
from .vispy_volume_layer import VispyVolumeLayer


def create_vispy_node(layer):
    """
    Create vispy visual node for a layer based on its layer type.

    Parameters
    ----------
        layer : napari.layers._base_layer.Layer
            Layer that needs its propetry widget created.

    Returns
    ----------
        node : vispy.scene.visuals.VisualNode
            Vispy visual node
    """
    name = 'Vispy' + type(layer).__name__ + 'Layer'
    try:
        VispyLayer = globals()[name]
        properties = VispyLayer(layer)
    except KeyError:
        properties = VispyBaseLayer(layer)

    return properties
