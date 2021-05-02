(intro-to-event-loop)=

# An Introduction to the Event Loop in napari


## tldr;

It is not necessary to have a deep understanding of Qt or event loops to use
napari. napari attempts to use "sane defaults" for most scenarios. Here are the
most important details:

### In IPython or Jupyter Notebook

napari will detect if you are running an an IPython or Jupyter shell, and will
automatically use the [IPython GUI event
loop](https://ipython.readthedocs.io/en/stable/config/eventloops.html#integrating-with-gui-event-loops).
As of [version 0.4.7](https://github.com/napari/napari/releases/tag/v0.4.7) is
no longer necessary to call `%gui qt` manually.  Just create a viewer:

```python
In [1]: import napari

In [2]: viewer = napari.Viewer()  # viewer will show in a new window

In [3]: ... # continue interactive usage
```

````{tip}
If you would prefer that napari did *not* start the interactive
event loop for you in IPython, then you can disable it with:

```python
from napari.utils import SETTINGS

SETTINGS.application.ipy_interactive = False
```

But then you will have to start the program yourself as described [below](#in-a-script).


````
### In a script

Outside of IPython, you must tell napari when to "start the program" using
{func}`napari.run`.  This will *block* execution of your script at that point,
show the viewer, and wait for any user interaction.  When the last viewer
closes, execution of the script will proceed.

```python
import napari

viewer = napari.Viewer()
... # continue setting  up your program

# start the program, show the viewer, wait for GUI interaction.
napari.run() 

# anything below here will execute only after the viewer is closed.
```


-----------


## More in depth...

Like most applications with a graphical user interface (GUI), napari operates
within an **event loop** that waits for – and responds to – events triggered by
the user interacting with the program.  These events might be something like a
mouse click, or a keypress, and usually correspond to some specific action taken
by the user (e.g. "user moved the gamma slider").

At its core, an event loop is rather simple.  It amounts to something that looks
like this (in pseudo-code):

```python
event_queue = Queue()

while True:  # infinite loop!
    if not event_queue.is_empty():
        event = get_next_event()
        if event.value == 'Quit':
            break
        else:
            process_event(event)
```

Actions taken by the user add events to the queue (e.g. "button pressed",
"slider moved", etc...), and the event loop handles them one at a time.

## Qt Applications and Event Loops

Currently, napari uses Qt as its GUI backend, and the main loop handling events
in napari is the [Qt
EventLoop](https://wiki.qt.io/Threads_Events_QObjects#Events_and_the_event_loop).

A deep dive into the Qt event loop is beyond the scope of this document, but
it's worth being aware of two critical steps in the "lifetime" of a Qt
Application:

1) Any program that would like to create a
   [`QWidget`](https://doc.qt.io/qt-5/qwidget.html) (the class from which all
   napari's graphical elements are subclassed), must create a
   [`QApplication`](https://doc.qt.io/qt-5/qapplication.html) instance *before*
   instantiating any widgets.

   ```python
   from qtpy.QtWidgets import QApplication

   app = QApplication([])
   ```

2) In order to actually show and interact with widgets, one must start the
   application's event loop:

   ```python
   app.exec_()
   ```

### napari's `QApplication`

In napari, the initial step of creating the `QApplication` is handled by
{func}`napari.qt.get_app`.  (Note however, that napari will do this for you
automatically behind the scenes when you create a viewer with
{class}`napari.Viewer()`)

The second step – starting the Qt event loop – is handled by {func}`napari.run`

```python
import napari

viewer = napari.Viewer()  # this will create a Application if one doesn't exist

napari.run()  # this will call `app.exec_()` and start the event loop.
```

(gui-qt-deprecated)=

:::{admonition}  What about `napari.gui_qt`? :class: tip

**{func}`napari.gui_qt` was deprecated in version 0.4.8.**

The autocreation of the `QApplication` instance and the {func}`napari.run`
function was introduced in
[PR#2056](https://github.com/napari/napari/pull/2056), and released in [version
0.4.3](https://github.com/napari/napari/releases/tag/v0.4.3).  Prior to that,
all napari examples included this `gui_qt()` context manager:

```python
# deprecated
with napari.gui_qt():
    viewer = napari.Viewer()
```

On entering the context, `gui_qt` would create a `QApplication`, and on exiting
the context, it would start the event loop (the two critical steps [mentioned
above](#qt-applications-and-event-loops)).  

Unlike a typical context manager, however, it did not actually *destroy* the
`QApplication` (since it may still be needed in the same session)... and future
calls to `gui_qt` were only needed to start the event loop.  By auto-creating
the `QApplication` during {class}`~napari.Viewer` creation, introducing the
explicit {func}`napari.run`, and using the [integrated IPython event
loop](#in-ipython-or-jupyter-notebook) when applicable, we hope to simplify the
usage of napari.

:::

## Hooking up your own events

If you're coming from a background of scripting or working with python in an
interactive console, thinking in terms of the "event loop" can feel a bit
strange at time.  Often we write code in a very procedural way: "do this ...
then do that, etc...". With napari and other GUI programs however, usually you
hook up a bunch of conditions and to callback functions (e.g. "If this event
happens, then call this function") and *then* start the loop and hope you hooked
everything up correctly!  Indeed, much of the ``napari`` source code is
dedicated to creating and handling events: search the codebase for [`.emit(`](https://github.com/napari/napari/search?q=%22.emit%28%22&type=code)
and [`.connect(`](https://github.com/napari/napari/search?q=%22.connect%28%22&type=code) to find examples of creating and handling internal events,
respectively.

If you would like to setup a custom event listener then you  need to hook into
the napari event.  We offer a couple of convenience decorators to easily connect
functions to key and mouse events.

### Listening for keypress events

One option is to use keybindings, that will listen for keypresses and then call
some callback whenever pressed, with the viewer instance passed as an argument
to that function. As a basic example, to add a random image to the viewer every
time the `i` key is pressed, and delete the last layer when the `k` key is
pressed:

```python
import numpy as np
import napari

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

napari.run()
```

See also this [custom key bindings
example](https://github.com/napari/napari/blob/master/examples/custom_key_bindings.py).

### Listening for mouse events

You can also listen for and react to mouse events, like a click or drag event,
as show here where we update the image with random data every time it is
clicked.

```python
import numpy as np
import napari

viewer = napari.Viewer()
layer = viewer.add_image(np.random.random((512, 512)))

@layer.mouse_drag_callbacks.append
def update_layer(layer, event):
    layer.data = np.random.random((512, 512))

napari.run()
```

See also the [custom mouse
functions](https://github.com/napari/napari/blob/master/examples/custom_mouse_functions.py)
and [mouse drag
callback](https://github.com/napari/napari/blob/master/examples/mouse_drag_callback.py)
examples.

### Connection functions to native napari events

If you want something to happen following some event that happens *within*
napari, then trick becomes knowing which native signals any given napari object
provides for you to "connect" to.  Until we have centralized documentation for
all of the events offered by napari objects, the best way to find these is to
browse the source code.  Take for instance, the base
{class}`~napari.layers.Layer` class: you'll find in the `__init__` method a
``self.events`` section that looks like this:

```python
self.events = EmitterGroup(
    ...
    data=Event,
    name=Event,
    ...
)
```

That tells you that all layers are capable of emitting events called `data`, and
`name` (among many others) that will (presumably) be emitted when that property
changes. To provide your own response to that change, you can hook up a callback
function that accepts the event object:

```python
def print_layer_name(event):
    print(f"{event.source.name} changed its data!")

layer.events.data.connect(print_layer_name)
```

## Long-running, blocking functions

An important detail here is that the napari event loop is running in a *single
thread*.  This works just fine if the handling of each event is very short, as
is usually the case with moving sliders, and pressing buttons.  However, if one
of the events in the queue takes a long time to process, then every other event
must wait!

Take this example in napari:

```python
viewer = napari.Viewer()
# everything is fine so far... but if we trigger a long computation
image = np.random.rand(512, 1024, 1024).mean(0)
viewer.add_image(image)
# the entire interface freezes!
```

Here we have a long computation (`np.random.rand(512, 1024, 1024).mean(0)`) that
"blocks" the main thread, meaning *no button press, key press, or any other
event can be processed until it's done*.  In this scenario, it's best to put
your long-running function into another thread or process.  `napari` provides a
convenience for that, described in {ref}`multithreading-in-napari`.
