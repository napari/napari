(multithreading-in-napari)=

# Multithreading in napari

As described in {ref}`intro-to-event-loop`, `napari`, like most GUI
applications, runs in an event loop that is continually receiving and
responding to events like button presses and mouse events.  This works fine
until one of the events takes a very long time to process.  A long-running
function (such as training a machine learning model or running a complicated
analysis routine) may "block" the event loop in the main thread, leading to a
completely unresponsive viewer.  The example used there was:

```python
import napari
import numpy as np


viewer = napari.Viewer()
# everything is fine so far... but if we trigger a long computation
image = np.random.rand(1024, 512, 512).mean(0)
viewer.add_image(image)
# the entire interface freezes!
```

In order to avoid freezing the viewer during a long-running blocking function,
you must run your function in another thread or process.

## Processes, threads, and `asyncio`

There are multiple ways to achieve "concurrency" (multiple things happening at
the same time) in python, each with their own advantages and disadvantages.
It's a rich, complicated topic, and a full treatment is well beyond the scope
of this document, but strategies generally fall into one of three camps:

1. Multithreading
2. Multprocessing
3. Single-thread concurrency with
   [asyncio](https://docs.python.org/3/library/asyncio.html)

For a good high level overview on concurrency in python, see 
[this post](https://realpython.com/python-concurrency/).
See the 
[trio docs](https://trio.readthedocs.io/en/stable/tutorial.html)
for a good introduction to Python's new `async/await` syntax.
And of course, see the python docs on
[threading](https://docs.python.org/3/library/threading.html),
[multiprocessing](https://docs.python.org/3/library/multiprocessing.html),
[concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html),
and [asyncio](https://docs.python.org/3/library/asyncio.html).

If you already have experience with any of these methods, you should be able to
immediately leverage them in napari.  `napari` also provides a few
convenience functions that allow you to easily run your long-running
methods in another thread.


## Threading in napari with `@thread_worker`

The simplest way to run a function in another thread in napari is to decorate
your function with the
{func}`@thread_worker <napari.qt.threading.thread_worker>` decorator.
Continuing with the example above:

```{code-block} python
---
emphasize-lines: 4,7,13-15
---
import napari
import numpy as np

from napari.qt.threading import thread_worker


@thread_worker
def average_large_image():
    return np.random.rand(1024, 512, 512).mean(0)

viewer = napari.Viewer()
worker = average_large_image()  # create "worker" object
worker.returned.connect(viewer.add_image)  # connect callback functions
worker.start()  # start the thread!
napari.run()
```

The {func}`@thread_worker <napari.qt.threading.thread_worker>` decorator
converts your function into one that returns a
{class}`~napari.qt.threading.WorkerBase` instance. The `worker`
manages the work being done by your function in another thread.  It also
exposes a few "signals" that let you respond to events happening in the other
thread.  Here, we connect the `worker.returned` signal to the
{meth}`viewer.add_image<napari.components.viewer_model.ViewerModel.add_image>`
function, which has the effect of adding the result to the viewer when it is
ready. Lastly, we start the worker with
{meth}`~napari.qt.threading.WorkerBase.start` because workers do not
start themselves by default.

The {func}`@thread_worker <napari.qt.threading.thread_worker>` decorator also
accepts keyword arguments like `connect`, and `start_thread`, which may
enable more concise syntax. The example below is equivalent to lines 7-15 in
the above example:

```python
viewer = napari.Viewer()

@thread_worker(connect={"returned": viewer.add_image})
def average_large_image():
    return np.random.rand(1024, 512, 512).mean(0)

average_large_image()
napari.run()
```

```{note}
When the `connect` argument to 
{func}`@thread_worker<napari.qt.threading.thread_worker>`
is not `None`, the thread will start
by default when the decorated function is called.  Otherwise the thread must
be manually started by calling
{meth}`worker.start() <napari.qt.threading.WorkerBase.start>`.
```

## Responding to feedback from threads

As shown above, the `worker` object returned by a function decorated with
{func}`@thread_worker <napari.qt.threading.thread_worker>` has a number of
signals that are emitted in response to certain events.  The base signals
provided by the `worker` are:

* `started` - emitted when the work is started
* `finished` - emitted when the work is finished
* `returned` [*value*] - emitted with return value when the function returns
* `errored` [*exception*] - emitted with an `Exception` object if an
  exception is raised in the thread.

### Example: Custom exception handler

Because debugging issues in multithreaded applications can be tricky, the
default behavior of a `@thread_worker` - decorated function is to re-raise
any exceptions in the main thread.  But just as we connected the
`worker.returned` event above to the `viewer.add_image` method, you can
also connect your own custom handler to the `worker.errored` event:

```python
def my_handler(exc):
    if isinstance(exc, ValueError):
        print(f"We had a minor problem {exc}")
    else:
        raise exc

@thread_worker(connect={"errored": my_handler})
def error_prone_function():
    ...
```

## Generators for the win!

````{admonition} quick reminder

A generator function is a
[special kind of function](https://realpython.com/introduction-to-python-generators/)
that returns a lazy iterator. To make a generator, you "yield" 
results rather than (or in addition to) "returning" them:

```python
def my_generator():
    for i in range(10):
        yield i
```
````

**Use a generator!** By writing our decorated function as a generator that
`yields` results instead of a function that `returns` a single result at
the end, we gain a number of valuable features, and a few extra signals and
methods on the `worker`.

* `yielded` [*value*]- emitted with a value when a value is yielded
* `paused` - emitted when a running job has successfully paused
* `resumed`  - emitted when a paused job has successfully resumed
* `aborted` - emitted when a running job is successfully aborted

Additionally, generator `workers` will also have a few additional methods:

* `send` - send a value *into* the thread (see below)
* `pause` - send a request to pause a running worker
* `resume` - send a request to resume a paused worker
* `toggle_pause` - send a request to toggle the running state of the worker
* `quit` - send a request to abort the worker

### Retrieving intermediate results

The most obvious benefit of using a generator is that you can monitor
intermediate results back in the main thread.  Continuing with our example of
taking the mean projection of a large stack, if we yield the cumulative average
as it is generated (rather than taking the average of the fully generated
stack) we can watch the mean projection as it builds:


```{code-block} python
---
emphasize-lines: 19,25
---
import napari
import numpy as np
from napari.qt.threading import thread_worker


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
napari.run()
```

Note how we periodically (every 16 iterations) `yield` the image result in
the `large_random_images` function.  We also connected the
`yielded` event in the
{func}`@thread_worker <napari.qt.threading.thread_worker>`
decorator to the previously-defined `update_layer` function.  The result is
that the image in the viewer is updated every time a new image is yielded.

Any time you can break up a long-running function into a stream of
shorter-running yield statements like this, you not only benefit from the
increased responsiveness in the viewer, you can often save on precious memory
resources.

#### Flow control and escape hatches

A perhaps even more useful aspect of yielding periodically in our long running
function is that we provide a "hook" for the main thread to control the flow of
our long running function.  When you use the
{func}`@thread_worker <napari.qt.threading.thread_worker>` decorator on a
generator function, the ability to stop, start, and quit a thread comes for
free.  In the example below we decorate what would normally be an infinitely
yielding generator, but add a button that aborts the worker when clicked:

```{code-block} python
---
emphasize-lines: 20,30
---
import time
import napari
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QPushButton

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

# add a button to the viewer that, when clicked, stops the worker
button = QPushButton("STOP!")
button.clicked.connect(worker.quit)
worker.finished.connect(button.clicked.disconnect)
viewer.window.add_dock_widget(button)

worker.start()
napari.run()
```

#### Graceful exit

A side-effect of this added flow control is that `napari` can gracefully
shutdown any still-running workers when you try to quit the program.  Try the
example above, but quit the program *without* pressing the "STOP" button.  No
problem!  `napari` asks the thread to stop itself the next time it yields,
and then closes without leaving any orphaned threads.

Now go back to the first example with the pure (non-generator) function, and
try quitting before the function has returned (i.e. before the image appears).
You'll notice that it takes a while to quit: it has to wait for the background
thread to finish because there is no good way to communicate the request that
it quit!  If you had a *very* long function, you'd be left with no choice but
to force quit your program.

So whenever possible, sprinkle your long-running functions with `yield`.

## Full two-way communication

So far we've mostly been *receiving* results from the threaded function, but we
can send values *into* a generator-based thread as well using
{meth}`worker.send() <napari.qt.threading.GeneratorWorker.send>` This works
exactly like a standard python 
[generator.send](https://docs.python.org/3/reference/expressions.html#generator.send)
pattern.  This next example ties together a number of concepts and demonstrates
two-thread communication with conditional flow control.  It's a simple
cumulative multiplier that runs in another thread, and exits if the product
hits "0":

```{code-block} python
---
emphasize-lines: 9,14-16,35,39,49,50,52,53
---
import napari
import time

from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QLineEdit, QLabel, QWidget, QVBoxLayout
from qtpy.QtGui import QDoubleValidator


@thread_worker
def multiplier():
    total = 1
    while True:
        time.sleep(0.1)
        new = yield total
        total *= new if new is not None else 1
        if total == 0:
            return "Game Over!"

viewer = napari.Viewer()

# make a widget to control the worker
# (not the main point of this example...)
widget = QWidget()
layout = QVBoxLayout()
widget.setLayout(layout)
result_label = QLabel()
line_edit = QLineEdit()
line_edit.setValidator(QDoubleValidator())
layout.addWidget(line_edit)
layout.addWidget(result_label)
viewer.window.add_dock_widget(widget)

# create the worker
worker = multiplier()

# define some callbacks
def on_yielded(value):
    worker.pause()
    result_label.setText(str(value))
    line_edit.setText('1')

def on_return(value):
    line_edit.setText('')
    line_edit.setEnabled(False)
    result_label.setText(value)

def send_next_value():
    worker.send(float(line_edit.text()))
    worker.resume()

worker.yielded.connect(on_yielded)
worker.returned.connect(on_return)
line_edit.returnPressed.connect(send_next_value)

worker.start()
napari.run()
```

Let's break it down:

1. As usual, we decorate our generator function with
   {func}`@thread_worker <napari.qt.threading.thread_worker>` and instantiate
   it to create a `worker`.

2. The most interesting line in this example is where we both
   `yield` the current ``total`` to the main thread (`yield total`), *and*
   receive a new value from the main thread (with `new = yield`).

3. In the main thread, we have connected that `worker.yielded` event
   to a callback that pauses the worker and updates the `result_label`
   widget.

4. The thread will then wait indefinitely for the `resume()` command,
   which we have connected to the `line_edit.returnPressed` signal.

5. However, before that `resume()` command gets sent, we use
   `worker.send()` to send the current value of the `line_edit` widget
   into the thread for multiplication by the existing total.

6. Lastly, if the thread total ever goes to "0", we stop the thread by
   returning the string ``"Game Over"``.  In the main thread, the
   `worker.returned` event is connected to a callback that disables the
   `line_edit` widget and shows the string returned from the thread.

This example is a bit contrived, since there's little need to put such a basic
computation in another thread.  But it demonstrates some of the power and
features provided when decorating a generator function with the
{func}`@thread_worker <napari.qt.threading.thread_worker>` decorator.

## Syntactic sugar

The {func}`@thread_worker <napari.qt.threading.thread_worker>` decorator is
just syntactic sugar for calling {func}`~napari.qt.threading.create_worker` on
your function.  In turn, {func}`~napari.qt.threading.create_worker` is just a
convenient "factory function" that creates the right subtype of `Worker`
depending on your function type. The following three examples are equivalent:

**Using the** `@thread_worker` **decorator:**

```python
from napari.qt.threading import thread_worker

@thread_worker
def my_function(arg1, arg2=None):
    ...

worker = my_function('hello', arg2=42)
```

**Using the** `create_worker` **function:**

```python
from napari.qt.threading import create_worker

def my_function(arg1, arg2=None):
    ...

worker = create_worker(my_function, 'hello', arg2=42)
```

**Using a** ``Worker`` **class:**

```python
from napari.qt.threading import FunctionWorker

def my_function(arg1, arg2=None):
    ...

worker = FunctionWorker(my_function, 'hello', arg2=42)
```

(the main difference between using `create_worker` and directly instantiating
the `FunctionWorker` class is that `create_worker` will automatically
dispatch the appropriate type of `Worker` class depending on whether the
function is a generator or not).

## Using a custom worker class

If you need even more control over the worker – such as the ability to define
custom methods or signals that the worker can emit, then you can subclass the
napari {class}`~napari.qt.threading.WorkerBase` class.  When doing so, please
keep in mind the following guidelines:

1. The subclass must either implement the
   {meth}`~napari.qt.threading.WorkerBase.work` method (preferred), or in
   extreme cases, may directly reimplement the
   {meth}`~napari.qt.threading.WorkerBase.run` method.  (When a worker "start"
   is started with {meth}`~napari.qt.threading.WorkerBase.start`, the call
   order is always
   {meth}`worker.start() <napari.qt.threading.WorkerBase.start>` →
   {meth}`worker.run() <napari.qt.threading.WorkerBase.run>` →
   {meth}`worker.work() <napari.qt.threading.WorkerBase.work>`.

2. When implementing the {meth}`~napari.qt.threading.WorkerBase.work` method,
   it is important that you periodically check `self.abort_requested` in your
   thread loop, and exit the thread accordingly, otherwise `napari` will not
   be able to gracefully exit a long-running thread.
   ```python
   def work(self):
       i = 0
       while True:
           if self.abort_requested:
               self.aborted.emit()
               break
               time.sleep(0.5)
    ```

3. It is also important to be mindful of the fact that the
   {meth}`worker.start() <napari.qt.threading.WorkerBase.start>` method adds
   the worker to a global Pool, such that it can request shutdown when exiting
   napari.  So if you re-implement `start`, please be sure to call
   `super().start()` to keep track of the `worker`.

4. When reimplementing the {meth}`~napari.qt.threading.WorkerBase.run` method,
   it is your responsibility to emit the `started`, `returned`,
   `finished`, and `errored` signals at the appropriate moments.

For examples of subclassing {class}`~napari.qt.threading.WorkerBase`, have a
look at the two main concrete subclasses in napari:
{class}`~napari.qt.threading.FunctionWorker` and
{class}`~napari.qt.threading.GeneratorWorker`.  You may also wish to simply
subclass one of those two classes.

### Adding custom signals

In order to emit signals, an object must inherit from `QObject`.  However,
due to challenges with multiple inheritance in Qt, the signals for
{class}`~napari.qt.threading.WorkerBase` objects actually live in the
`WorkerBase._signals` attribute (though they are accessible directly in the
worker namespace).  To add custom signals to a
{class}`~napari.qt.threading.WorkerBase` subclass you must first create a new
`QObject` with signals as class attributes:

```python
from qtpy.QtCore import QObject, Signal

class MyWorkerSignals(QObject):
    signal_name = Signal()

# or subclass one of the existing signals objects to "add"
# additional signals:

from napari.qt.threading import WorkerBaseSignals

# WorkerBaseSignals already has started, finished, errored...
class MyWorkerSignals(WorkerBaseSignals):
    signal_name = Signal()
```

and then either directly override the `self._signals` attribute on the
{class}`~napari.qt.threading.WorkerBase` class with an instance of your
signals class:


```python
class MyWorker(WorkerBase):

    def __init__(self):
        super().__init__()
        self._signals = MyWorkerSignals()
```

... or pass the signals class as the `SignalsClass` argument when
initializing the superclass in your `__init__` method:

```python
class MyWorker(WorkerBase):

    def __init__(self):
        super().__init__(SignalsClass=MyWorkerSignals)
```
