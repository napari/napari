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
        return np.random.rand(512, 1024, 1024).mean(0)

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
            return np.random.rand(512, 1024, 1024).mean(0)

        average_large_image()


Responding to feedback from threads
-----------------------------------

As shown above, the ``worker`` object returned by a function decorated with
``@thread_worker`` has a number of signals that are emitted in response to
certain events.  The base signals provided by the ``worker`` are:

* **started** - emitted when the work is started
* **finished** - emitted when the work is finished
* **returned** [*value*] - emitted with return value when the function returns
* **errored** [*exception*] - emitted with an ``Exception`` object if an
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

**Use a generator!**

By writing our decorated function as a generator, we gain a number of very
valuable features.

Intermediate Results
^^^^^^^^^^^^^^^^^^^^

The most obvious benefit is that you can "peek" at intermediate results back in
the main thread.  Continuing with our example of taking the mean projection of
very large stack, if we yield each plane as it is generated, we can watch the
mean projection as it builds:





Syntactic sugar
---------------

The ``@thread_worker`` decorator is just syntactic sugar for calling the 
``create_worker`` function on your function.  And in turn, ``create_worker`` is
just a convenience that creates the right type of ``Worker`` depending on your
function type. The following three examples are equivalent:

.. code-block:: python

    # with `@thread_worker` decorator
    from napari._qt.threading import thread_worker

    @thread_worker
    def my_function(arg1, arg2=None):
        pass

    worker = my_function('hello', arg2=42)


    # with `create_worker`
    from napari._qt.threading import create_worker

    def my_function(arg1, arg2=None):
       pass

    worker = create_worker(my_function, 'hello', arg2=42)



    # with Worker class
    from napari._qt.threading import FunctionWorker
    
    def my_function(arg1, arg2=None):
       pass

    worker = FunctionWorker(my_function, 'hello', arg2=42)

(the main difference between using ``create_worker`` and directly instantiating
the ``FunctionWorker`` class is that ``create_worker`` will automatically
dispatch the appropriate type of ``Worker`` class depending on whether the
function is a generator or not).