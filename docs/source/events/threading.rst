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

In order to avoid freezing the viewer during a long-running blocking function,
you must run your function in another thread or process.

Processes, Threads, and ``asyncio``
-----------------------------------

There are multiple ways to achieve "concurrency" (multiple things happening at
the same time) in python, each with their own advantages and disadvantages.
It's a rich, complicated topic, and a full treatment is well beyond the scope
of this document, but strategies generally fall into one of three camps:

1. Multithreading
2. Multprocessing
3. Single-thread concurrency with `asyncio
   <https://docs.python.org/3/library/asyncio.html>`_

For a good high level overview, see `this post
<https://realpython.com/python-concurrency/>`_.  For details, see the
python docs on `threading <https://docs.python.org/3/library/threading.html>`_,
`multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`_,
`concurrent.futures <https://docs.python.org/3/library/concurrent.futures.html>`_,
and `asyncio <https://docs.python.org/3/library/asyncio.html>`_

If you already have experience with any of these methods, you should be able to
immediately leverage them in napari.  ``napari`` also provides a few
convenience functions that allow you to easily run your long-running
methods in another thread.


Threading in napari with ``@thread_worker``
-------------------------------------------

The simplest way to run a function in another thread in napari is to decorate
your function with the ``@thread_worker`` decorator.  Taking the example above:

.. code-block:: python


