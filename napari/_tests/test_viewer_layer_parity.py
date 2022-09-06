"""
Ensure that layers and their convenience methods on the viewer
have the same signatures.
"""

import inspect

from napari import Viewer
from napari.layers import Image
from napari.view_layers import imshow


def test_signature_imshow():
    layer = Image
    name = layer.__name__
    method = imshow

    # collect the signatures for this method and its classes
    class_parameters = {
        **inspect.signature(layer.__init__).parameters,
        **inspect.signature(Viewer.__init__).parameters,
    }
    method_parameters = dict(inspect.signature(method).parameters)

    # Remove unique parameters from viewer method
    del method_parameters['viewer']  # only in this method
    del method_parameters[
        'channel_axis'
    ]  # gets added in viewer_model (TODO: I don't understand this)
    del class_parameters['self']  # only in class

    # ensure both have the same parameters
    assert set(class_parameters) == set(method_parameters)

    # ensure the parameters have the same defaults
    for name, parameter in class_parameters.items():
        fail_msg = f'Signature mismatch on {parameter}'
        assert method_parameters[name].default == parameter.default, fail_msg
