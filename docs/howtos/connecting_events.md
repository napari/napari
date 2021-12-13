(connecting-events)=

# Hooking up your own events

If you're coming from a background of scripting or working with python in an
interactive console, thinking in terms of the "event loop" can feel a bit
strange at times. Often we write code in a very procedural way: "do this ...
then do that, etc...". With napari and other GUI programs however, usually you
hook up a bunch of conditions to callback functions (e.g. "If this event
happens, then call this function") and *then* start the loop and hope you hooked
everything up correctly!  Indeed, much of the ``napari`` source code is
dedicated to creating and handling events: search the codebase for
[`.emit(`](https://github.com/napari/napari/search?q=%22.emit%28%22&type=code)
and
[`.connect(`](https://github.com/napari/napari/search?q=%22.connect%28%22&type=code)
to find examples of creating and handling internal events, respectively.

If you would like to set up a custom event listener then you need to hook into
the napari event loop. We offer a couple of convenience decorators to easily
connect functions to key and mouse events.

## Listening for keypress events

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
example](https://github.com/napari/napari/blob/main/examples/custom_key_bindings.py).

## Listening for mouse events

You can also listen for and react to mouse events, like a click or drag event,
as shown here where we update the image with random data every time it is
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

As of this writing `MouseProvider`s have 4 list of callbacks that can be registered:

   - `mouse_move_callbacks`
   - `mouse_wheel_callbacks`
   - `mouse_drag_callbacks`
   - `mouse_double_click_callbacks`

Please look at the documentation of `MouseProvider` for a more in depth
discussion of when each callback is triggered. In particular single click can be
registered with `mouse_drag_callbacks`, and `mouse_double_click_callbacks` is
triggered _in addition to_ mouse `mouse_drag_callbacks`.

See also the [custom mouse
functions](https://github.com/napari/napari/blob/main/examples/custom_mouse_functions.py)
and [mouse drag
callback](https://github.com/napari/napari/blob/main/examples/mouse_drag_callback.py)
examples.

## Connection functions to native napari events

If you want something to happen following some event that happens *within*
napari, the trick becomes knowing which native signals any given napari object
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
