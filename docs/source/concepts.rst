Concepts in napari
==================

The Viewer
----------

At the heart of napari is the viewer model.  For a tutorial introduction to the
viewer, see the `napari viewer tutorial <https://napari.org/tutorials/viewer>`_.

The :class:`~napari.viewer.Viewer` object contains, among other things:

- the main canvas
- an instance of :class:`~napari.components.layerlist.LayerList` - a list of
  layer objects that have been added to the :class:`~napari.viewer.Viewer`
