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
       image = np.random.rand(1024, 512, 512).mean(0)
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
your function with the ``@thread_worker`` decorator.  Continuing with the
example above:

.. code-block:: python
   :linenos:
   :emphasize-lines: 4,7,13-15

    import napari
    import numpy as np

    from napari._qt.threading import thread_worker


    @thread_worker
    def average_large_image():
        return np.random.rand(1024, 512, 512).mean(0)

    with napari.gui_qt():
        viewer = napari.Viewer()
        worker = average_large_image()  # create "worker" object
        worker.returned.connect(viewer.add_image)  # connect callback functions
        worker.start()  # start the thread!


The ``@thread_worker`` decorator (**7**), converts your function into one that
returns a ``worker`` instance (**13**). The ``worker`` manages the work being
done by your function in another thread.  It also exposes a few "signals" that
let you respond to events happening in the other thread.  Here, we connect the
``worker.returned`` signal to the ``viewer.add_image`` function (**14**), which
has the effect of adding the result to the viewer when it is ready. Lastly, we
start the worker with ``worker.start()`` (**15**) because workers do not start
themselves by default.

The ``@thread_worker`` decorator also accepts keyword arguments like
``connect``, and ``start_thread``, which may enable more concise syntax.
The example below is equivalent to lines 7-15 in the above example:

.. code-block:: python

    with napari.gui_qt():
        viewer = napari.Viewer()

        @thread_worker(connect={"returned": viewer.add_image}, start_thread=True)
        def average_large_image():
            return np.random.rand(1024, 512, 512).mean(0)

        average_large_image()


Responding to feedback from threads
-----------------------------------

As shown above, the ``worker`` object returned by a function decorated with
``@thread_worker`` has a number of signals that are emitted in response to
certain events.  The base signals provided by the ``worker`` are:

* ``started`` - emitted when the work is started
* ``finished`` - emitted when the work is finished
* ``returned`` [*value*] - emitted with return value when the function returns
* ``errored`` [*exception*] - emitted with an ``Exception`` object if an
  exception is raised in the thread.

Example: custom exception handler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because debugging issues in multithreaded applications can be tricky, the
default behavior of a ``@thread-worker`` - decorated function is to re-raise
any exceptions in the main thread.  But just as we connected the
``worker.returned`` event above to the ``viewer.add_image`` method, you can
also connect your own custom handler to the ``worker.errored`` event:

.. code-block:: python

    def my_handler(exc):
        if isinstance(exc, ValueError):
            print(f"We had a minor problem {exc}")
        else:
            raise exc

   @thread_worker(connect={"errored": my_handler})
    def error_prone_function():
        ...


Generators for the win!
-----------------------

.. admonition::  quick reminder

   A generator function is a `special kind of function
   <https://realpython.com/introduction-to-python-generators/>`_ that returns
   a lazy iterator. To make a generator, you "yield" results rather than (or in
   addition to) "returning" them:

   .. code-block:: python

        def my_generator():
            for i in range(10):
                yield i
        

**Use a generator!** By writing our decorated function as a generator that
``yields`` results instead of a function that ``returns`` a single result at
the end, we gain a number of valuable features, and a few extra signals and
methods on the ``worker``.

* ``yielded`` [*value*]- emitted with a value when a value is yielded
* ``paused`` - emitted when a running job has successfully paused
* ``resumed``  - emitted when a paused job has successfully resumed
* ``aborted`` - emitted when a running job is successfully aborted

Additionally, generator ``workers`` will also have a few additional methods:

* ``send`` - send a value *into* the thread (see below)
* ``toggle_pause`` - toggle the running state of the worker
* ``quit`` - send a request to abort the worker


Retreiving Intermediate Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most obvious benefit of using a generator is that you can monitor
intermediate results back in the main thread.  Continuing with our example of
taking the mean projection of a large stack, if we yield the cumulative average
as it is generated (rather than taking the average of the fully generated
stack) we can watch the mean projection as it builds:


.. code-block:: python
   :linenos:
   :emphasize-lines: 14,20

    with napari.gui_qt():
        viewer = napari.Viewer()

        def update_layer(new_image):
            try:
                # if the layer exists, update the data
                viewer.layers['result'].data = new_image
            except KeyError:
                # otherwise add it to the viewer
                viewer.add_image(
                    new_image, contrast_limits=(0.45, 0.55), name='result'
                )

        @thread_worker(connect={'yielded': update_layer})
        def large_random_images():
            cumsum = np.zeros((512, 512))
            for i in range(1024):
                cumsum += np.random.rand(512, 512)
                if i % 16 == 0:
                    yield cumsum / (i + 1)

        large_random_images()  # call the function!

Note how we periodically (every 16 iterations) ``yield`` the image result in
the ``large_random_images`` function (**20**).  We also connected the
``yielded`` event in the ``@thread_worker`` decorator to the previously-defined
``update_layer`` function (**14**).  The result is that the image in the viewer
is updated everytime a new image is yielded.

Any time you can break up a long-running function into a stream of
shorter-running yield statements like this, you not only benefit from the
increased responsivity in the viewer, you can often save on precious memory
resources.


Flow control and escape hatches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A perhaps even more useful aspect of yielding periodically in our long running
function is that we provide a "hook" for the main thread to control the flow
of our long running function.  When you use the ``@thread_worker`` decorator on
a generator function, the ability to stop, start, and quit a thread comes for
free.  In the example below we decorate what would normally be an infinitely
yielding generator, but add a button that aborts the worker when clicked:

.. code-block:: python
   :linenos:
   :emphasize-lines: 19,28
    
    import time
    import napari
    from qtpy.QtWidgets import QPushButton

    with napari.gui_qt():
        viewer = napari.Viewer()

        def update_layer(new_image):
            try:
                viewer.layers['result'].data = new_image
            except KeyError:
                viewer.add_image(
                    new_image, name='result', contrast_limits=(-0.8, 0.8)
                )

        @thread_worker
        def yield_random_images_forever():
            i = 0
            while True:  # infinite loop!
                yield np.random.rand(512, 512) * np.cos(i * 0.2)
                i += 1
                time.sleep(0.05)

        worker = yield_random_images_forever()
        worker.yielded.connect(update_layer)

        # add a button to the viewew that, when clicked, stops the worker
        button = QPushButton("STOP!")
        button.clicked.connect(worker.quit)
        worker.finished.connect(button.clicked.disconnect)
        viewer.window.add_dock_widget(button)

        worker.start()

Graceful exit
^^^^^^^^^^^^^

A side-effect of this added flow control is that ``napari`` can gracefully
shutdown any still-running workers when you try to quit the program.  Try the
example above, but quit the program *without* pressing the "STOP" button.  No
problem!  ``napari`` asks the thread to stop itself the next time it yields,
and then closes without leaving any orphaned threads.

Now go back to the first example with the pure (non-generator) function, and
try quitting before the function has returned (i.e. before the image appears).
You'll notice that it takes a while to quit: it has to wait for the background
thread to finish because there is no good way to communicate equest that it
quit!  If you had a *very* long function, you'd be left with no choice but to
force quit your program.

So whenever possible, sprinkle your long-running functions with ``yield``.

Two-way communication
---------------------

So far we've mostly been *receiving* results from the threaded function, but we
can send values into the thread as well using ``worker.send``.  This works
exactly like a standard python `generator.send
<https://docs.python.org/3/reference/expressions.html#generator.send>`_ 
pattern.



Syntactic sugar
---------------

The ``@thread_worker`` decorator is just syntactic sugar for calling 
``create_worker`` on your function.  In turn, ``create_worker`` is just a
convenient "factory function" that creates the right type of ``Worker``
depending on your function type. The following three examples are equivalent:

**Using the** ``@thread_worker`` **decorator:**

.. code-block:: python

    from napari._qt.threading import thread_worker

    @thread_worker
    def my_function(arg1, arg2=None):
        ...

    worker = my_function('hello', arg2=42)

**Using the** ``create_worker`` **function:**

.. code-block:: python

    from napari._qt.threading import create_worker

    def my_function(arg1, arg2=None):
       ...

    worker = create_worker(my_function, 'hello', arg2=42)

**Using a** ``Worker`` **class:**

.. code-block:: python

    from napari._qt.threading import FunctionWorker
    
    def my_function(arg1, arg2=None):
       ...

    worker = FunctionWorker(my_function, 'hello', arg2=42)

(the main difference between using ``create_worker`` and directly instantiating
the ``FunctionWorker`` class is that ``create_worker`` will automatically
dispatch the appropriate type of ``Worker`` class depending on whether the
function is a generator or not).