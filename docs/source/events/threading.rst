.. _multithreading-in-napari:

Multithreading in napari
========================

As described in :ref:`intro-to-event-loop`, ``napari``, like most GUI
applications, runs in an event loop that is continually receiving and
responding to events like button presses and mouse events.  This works fine
until one of the events takes a very long time to process.  A long-running
function (such as training a machine learning model or running a complicated
analysis routine) may "block" the event loop in the main thread, leading to a
completely unresponsive viewer.  The example used there was:

.. code-block:: python

   import napari
   import numpy as np

   with napari.gui_qt():
       viewer = napari.Viewer()
       # everything is fine so far... but if we trigger a long computation
       image = np.random.rand(512, 1024, 1024).mean(0)
       viewer.add_image(image)
       # the entire interface freezes!
