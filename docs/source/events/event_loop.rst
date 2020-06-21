.. _intro-to-event-loop:

An Introduction to the Event Loop in napari
===========================================

Like most applications with a graphical user interface (GUI), napari operates
within an **event loop** that waits for – and responds to – events triggered by
the user interacting with the program.  These events might be something like a
mouse click, or a keypress, and usually correspond to some specific action
taken by the user (e.g. "user moved the gamma slider").

At its core, an event loop is rather simple.  It amounts to something
that looks like this (in pseudo-code):

.. code-block:: python
   
    event_queue = Queue()

    while True:  # infinite loop!
        if not event_queue.is_empty():
            event = get_next_event()
            if event.value == 'Quit':
                break
            else:
                process_event(event)

Actions taken by the user add events to the queue ("button pressed",
"slider moved", etc...), and the event loop handles them one at a time. 

The Qt Event Loop
-----------------

Currently, napari uses Qt as its GUI backend, and the main loop handling events
in napari is the `Qt EventLoop
<https://wiki.qt.io/Threads_Events_QObjects#Events_and_the_event_loop>`_.
When you use the following syntax:

.. code-block:: python

    with napari.gui_qt():
        viewer = napari.Viewer()

... you are starting up the Qt event loop.  This also explains why the only
wait to get *out* of that ``gui_qt`` context is to *stop* the Qt event loop
(usually by quitting the napari viewer).  A deep dive into the Qt event loop is
beyond the scope of this document, but it's worth being aware of the central role
that it plays in napari, and users interested in creating highly customized
events and actions are advised to gain at least a little familiarity with the
Qt event loop.


Hooking up your own events
--------------------------

If you're coming from a background of scripting or working with python in an
interactive console, thinking in terms of the "event loop" can feel a bit
strange at time.  Often we write code in a very procedural way: "do this ...
then do that, etc...". With napari and other GUI programs however, usually you
hook up a bunch of conditions and to callback functions (e.g. "If this event
happens, then call this function") and *then* start the loop and hope you
hooked everything up correctly!  Indeed, much of the ``napari`` source code is
dedicated to creating and handling events: search the codebase for "``.emit(``"
and "``.connect(``" to find examples of creating and handling internal events,
respectively.

If you would like to setup a custom event listener then you  need to hook into
the napari event.  We offer a couple of convenience decorators to easily
connect functions to key and mouse events.

Listening for keypress events
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One option is to use keybindings, that will listen for keypresses and then call
some callback whenever pressed, with the viewer instance passed as an argument
to that function. As a basic example, to add a random image to the viewer
every time the ``i`` key is pressed, and delete the last layer when the ``k``
key is pressed:

.. code-block:: python

    import numpy as np
    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()

        @viewer.bind_key('i')
        def add_layer(viewer):
            viewer.add_image(np.random.random((512, 512)))

        @viewer.bind_key('k')
        def delete_layer(viewer):
            try:
                viewer.layers.pop(0)
            except IndexError:
                pass

See also this `custom key bindings example
<https://github.com/napari/napari/blob/master/examples/custom_key_bindings.py>`_

Listening for mouse events
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also listen for and react to mouse events, like a click or drag event,
as show here where we update the image with random data every time it is
clicked.

.. code-block:: python

    import numpy as np
    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        layer = viewer.add_image(np.random.random((512, 512)))

        @layer.mouse_drag_callbacks.append
        def update_layer(layer, event):
            layer.data = np.random.random((512, 512))

See also the `custom mouse functions
<https://github.com/napari/napari/blob/master/examples/custom_mouse_functions.py>`_
and `mouse drag callback
<https://github.com/napari/napari/blob/master/examples/mouse_drag_callback.py>`_
examples

Connection functions to native napari events
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want something to happen following some event that happens *within*
napari, then trick becomes knowing which native signals any given napari object
provides for you to "connect" to.  Until we have centralized documentation for
all of the events offered by napari objects, the best way to find these is to
browse the source code.  Take for instance, the base
:class:`~napari.layers.base.base.Layer` class: you'll find in the ``__init__``
method a ``self.events`` section that looks like this:

.. code-block:: python

    self.events = EmitterGroup(
        ...
        data=Event,
        name=Event,
        ...
    )

That tells you that all layers are capable of emitting events called ``data``,
and ``name`` (among many others) that will (presumably) be emitted when that
property changes. To provide your own response to that change, you can hook up
a callback function that accepts the event object:

.. code-block:: python

    def print_layer_name(event):
        print(f"{event.source.name} changed its data!")

    layer.events.data.connect(print_layer_name)


Long-running, blocking functions
--------------------------------

An important detail here is that the napari event loop is running in a *single
thread*.  This works just fine if the handling of each event is very short, as
is usually the case with moving sliders, and pressing buttons.  However, if one
of the events in the queue takes a long time to process, then every other event
must wait!

Take this example in napari:

.. code-block:: python

    import napari
    import numpy as np

    with napari.gui_qt():
        viewer = napari.Viewer()
        # everything is fine so far... but if we trigger a long computation
        image = np.random.rand(512, 1024, 1024).mean(0)
        viewer.add_image(image)
        # the entire interface freezes!

Here we have a long computation (``np.random.rand(512, 1024, 1024).mean(0)``)
that "blocks" the main thread, meaning *no button press, key press, or any
other event can be processed until it's done*.  In this scenario, it's best to
put your long-running function into another thread or process.  ``napari``
provides a convenience for that, described in :ref:`multithreading-in-napari`.