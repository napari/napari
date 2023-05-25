"""
Ensure that layers and their convenience methods on the viewer
have the same signatures.
"""

import inspect

from napari import Viewer
from napari.view_layers import imshow


def test_imshow_signature_consistency():
    # Collect the signatures for imshow and the associated Viewer methods
    viewer_parameters = {
        **inspect.signature(Viewer.__init__).parameters,
        **inspect.signature(Viewer.add_image).parameters,
    }
    imshow_parameters = dict(inspect.signature(imshow).parameters)

    # Remove unique parameters
    del imshow_parameters['viewer']
    del viewer_parameters['self']

    # Ensure both have the same parameter names
    assert imshow_parameters.keys() == viewer_parameters.keys()

    # Ensure the parameters have the same defaults
    for name, parameter in viewer_parameters.items():
        # data is a required for imshow, but optional for add_image
        if name == 'data':
            continue
        fail_msg = f'Signature mismatch on {parameter}'
        assert imshow_parameters[name].default == parameter.default, fail_msg
