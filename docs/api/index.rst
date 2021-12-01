.. _api:

API Reference
=============

Information on specific functions, classes, and methods.


Modules
-------

.. rubric:: Primary

For the average user's workflows.

.. autosummary::
   :toctree:
   :recursive:

   napari.layers
   napari.view_layers
   napari.types
   napari.utils

.. rubric:: Advanced

For those wishing to add custom functionality to their project.

.. autosummary::
   :toctree:
   :recursive:

   napari.plugins
   napari.components
   napari.qt.threading
   napari.utils.perf


.. autosummary::
   :toctree:

   napari

Starting the Event Loop
-----------------------

.. autosummary::
   napari.gui_qt
   napari.run


Viewing a Layer
---------------

.. autosummary::
   napari.view_image
   napari.view_labels
   napari.view_path
   napari.view_points
   napari.view_shapes
   napari.view_surface
   napari.view_tracks
   napari.view_vectors

.. autosummary:: napari.Viewer

Misc
----

.. autosummary::
   napari.save_layers
   napari.sys_info
