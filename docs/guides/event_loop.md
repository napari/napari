(intro-to-event-loop)=

# An introduction to the event loop in napari

## Brief summary

It is not necessary to have a deep understanding of Qt or event loops to use
napari. napari attempts to use "sane defaults" for most scenarios. Here are the
most important details:

### In IPython or Jupyter Notebook

napari will detect if you are running an IPython or Jupyter shell, and will
automatically use the [IPython GUI event
loop](https://ipython.readthedocs.io/en/stable/config/eventloops.html#integrating-with-gui-event-loops).
As of [version 0.4.7](https://github.com/napari/napari/releases/tag/v0.4.7), it is
no longer necessary to call `%gui qt` manually.  Just create a viewer:

```python
In [1]: import napari

In [2]: viewer = napari.Viewer()  # Viewer will show in a new window

In [3]: ... # Continue interactive usage
```

````{tip}
If you would prefer that napari did *not* start the interactive
event loop for you in IPython, then you can disable it with:

```python
from napari.settings import get_settings

get_settings().application.ipy_interactive = False
```

... but then you will have to start the program yourself as described [below](#in-a-script).
````

### In a script

Outside of IPython, you must tell napari when to "start the program" using
{func}`napari.run`.  This will *block* execution of your script at that point,
show the viewer, and wait for any user interaction.  When the last viewer
closes, execution of the script will proceed.

```python
import napari

viewer = napari.Viewer()
...  # Continue setting  up your program

# Start the program, show the viewer, wait for GUI interaction.
napari.run() 

# Anything below here will execute only after the viewer is closed.
```

-----------

## More in depth...

Like most applications with a graphical user interface (GUI), napari operates
within an **event loop** that waits for – and responds to – events triggered by
the user's interactions with the interface.  These events might be something like a
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

## Qt applications and event loops

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

   app = QApplication([])  # where [] is a list of args passed to the App
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

viewer = napari.Viewer()  # This will create a Application if one doesn't exist

napari.run()  # This will call `app.exec_()` and start the event loop.
```

(gui-qt-deprecated)=

:::{admonition}  What about `napari.gui_qt`?
:class: caution

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
explicit {func}`napari.run` function, and using the [integrated IPython event
loop](#in-ipython-or-jupyter-notebook) when applicable, we hope to simplify the
usage of napari.

:::

Now that you have an understanding of how napari creates the event loop, you may
wish to learn more about {ref}`hooking up your own actions <connecting-events>` and callbacks to specific events.
