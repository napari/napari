from napari.layers import (
    Image,
    Pyramid,
    Markers,
    Labels,
    Shapes,
    Vectors,
)
from napari import Viewer

def test_docstrings():
    for obj in (Image, Pyramid, Markers, Labels, Shapes, Vectors):
        method = f'add_{obj.__name__.lower()}'
        obj_doc = obj.__doc__
        # We only check the parameters section, which is the first
        # section after separating by empty line. We also strip empty
        # line and white space that occurs at the end of a class docstring
        obj_param = obj_doc.split('\n\n')[1].rstrip('\n ')
        method_doc = getattr(Viewer, method).__doc__
        # For the method docstring, we also need to
        # remove the extra indentation of the method docstring compared
        # to the class
        method_param = method_doc.split('\n\n')[1].replace('\n    ', '\n')[4:]
        fail_msg = f"Docstrings don't match for class {obj.__name__}"
        assert obj_param == method_param, fail_msg
