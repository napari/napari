.. _hook-specifications-reference:

napari hook specification reference
===================================

.. automodule:: napari.plugins.hook_specifications
  :noindex:

.. currentmodule:: napari.plugins.hook_specifications

IO hooks
--------

.. autofunction:: napari_provide_sample_data
.. autofunction:: napari_get_reader
.. autofunction:: napari_get_writer

.. _write-single-layer-hookspecs:

Single Layers IO
''''''''''''''''

The following hook specifications will be called when a user saves a single
layer in napari, and should save the layer to the requested format and return
the save path if the data are successfully written. Otherwise, if nothing was saved, return ``None``.
They each accept a ``path``.
It is up to plugins to inspect and obey the extension of the path (and return
``False`` if it is an unsupported extension).  The ``data`` argument will come
from ``Layer.data``, and a ``meta`` dict that will correspond to the layer's
:meth:`~napari.layers.base.base.Layer._get_state` method.

.. autofunction:: napari_write_image
.. autofunction:: napari_write_labels
.. autofunction:: napari_write_points
.. autofunction:: napari_write_shapes
.. autofunction:: napari_write_surface
.. autofunction:: napari_write_vectors

Analysis hooks
--------------

.. autofunction:: napari_experimental_provide_function

GUI hooks
---------

.. autofunction:: napari_experimental_provide_dock_widget
