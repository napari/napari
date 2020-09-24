.. _perfmon:

Performance Monitoring
======================

If napari is not performing well, you can use the
:mod:`napari.utils.perf<napari.utils.perf>` to help
diagnose the problem. The module can do several things:

1. Time Qt Events 

2. Display a dockable **performance** widget.

3. Write JSON trace files viewable with `chrome://tracing`.

4. Time any function or method you specify in the config file.

Monitoring vs. Profiling
------------------------

Profiling is similar to performance monitoring. Profiling usually
involves running an external tool to acquire timing data on
every function and method in the program. Sometimes this will
cause the program to run so slowly it's hard to use the program
interactively.

Performance monitoring does not require running a separate tool to collect
the timing information. Although we do use Chrome to view the trace files.
If you don't time too many functions and and methods, the program can run
at close to full speed.

This document discusses napari's Performance Monitoring features. Profiling
napari might be useful as well, but it is not discussed here.


Enabling perfmon
----------------

There are two ways to enable performance monitoring. Set the environment
variable `NAPARI_PERFMON=1` or set `NAPARI_PERFMON` to the path of 
a JSON configuration file, for example `NAPARI_PERFMON=/tmp/perfmon.json`.

Setting `NAPARI_PERFMON=1` does three things:

1. Time Qt Events
2. Show the dockable **performance** widget.
3. Reveal the **Debug** menu which you can use to create a trace file.


Configuration File Format
-------------------------

Example configuration file:

.. code-block:: json

    {
        "trace_qt_events": true,
        "trace_file_on_start": "/tmp/latest.json",
        "trace_callables": [
            "chunk_loader"
        ],
        "callable_lists": {
            "chunk_loader": [
                "napari.components.chunk._loader.ChunkLoader.load_chunk",
                "napari.components.chunk._loader.ChunkLoader._done"
            ]
        }
    }


Configuration Options
---------------------

`trace_qt_events` 
~~~~~~~~~~~~~~~~~

If true perfmon will time the duration of all Qt Events. You might
want to turn this off if the overhead is noticeable, or if you want
your trace file to be less cluttered.

`trace_file_on_start`
~~~~~~~~~~~~~~~~~~~~~

If a path is given, napari will start tracing immediately on start. In many
cases this is much more convenient than using the Debug Menu. Be sure to
exit napari using the **Quit** command. The trace file will be written on
exit.

`trace_callables`
~~~~~~~~~~~~~~~~~

Specify which `callable_lists` you want to trace. You can have many
`callable_lists` defined, but this setting says which should be traced.

`callable_lists`
~~~~~~~~~~~~~~~~

These lists can be referenced by the `callable_lists` option. For example
you might want one list for "rendering" and another for "painting".

Trace File
-----------

The trace file that napari produces is viewable in Chrome. Go to the
special URL `chrome://tracing` in Chrome. Use the **Load** button inside
the Chrome window, or just drag-n-drop your JSON trace file into the Chrome
window.

You can also view trace files using the Speedscope website at
https://www.speedscope.app/. It is similar to `chrome://tracing` but has
some different features.

The trace file format is specifed in the [Trace File Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview)
Google Doc. The format is well-documented, but there are no pictures so
it's not always clear how a given feature actually looks in the Chrome
Tracing GUI.

Example Investigation
---------------------

Add a Sleep
~~~~~~~~~~~

To simulate part of napari running slow, add a `sleep()` call to the
:meth:`Labels.paint<napari.layer.labels.Label.paint>` method, this 
will make the method take at least 100ms:

.. code-block:: python
   :emphasize-lines: 2-3

    def paint(self, coord, new_label, refresh=True):
        import time
        time.sleep(0.1)

        if refresh is True:
            self._save_history()


Create a Perfmon Config File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a minimal perfmon config file `/tmp/perfmon.json` like this:

.. code-block:: json

    {
        "trace_qt_events": true,
        "trace_file_on_start": "/tmp/latest.json",
        "trace_callables": []
    }

This will write `/tmp/latest.json` every time we run napari and then exit
with the **Quit** commmand. This is often easier than manually start a trace
using the **Debug** menu. 


Run napari
~~~~~~~~~~

Now run napari's `add_labels` example like this:

.. code-block:: shell

    NAPARI_PERFMON=/tmp/perfmon.json python examples/add_labels.py

Use the paint tool to draw on the labels layer. Notice in the
**performance** widget it prints some events took over 100ms. It says both
events took over 100ms, but really one probably called the other one, the
call hierarchy is not shown:

.. image:: https://user-images.githubusercontent.com/4163446/94198620-898c4c00-fe85-11ea-8769-83f52c0a1aad.png

Exit napari using the **Quit** command so that it writes the trace file on exit.

View Trace in Chrome
~~~~~~~~~~~~~~~~~~~~

Run Chrome and go to the URL `chrome://tracing`. Drag and drop
`/temp/latest.json` into the Chrome window. You can navigate multiple ways,
but one easy way is use the `AD` keys to move left and right, and use `WS`
keys to zoom in or out.

Locate one of the slow `MouseMove` events and click on it. In the lower
pane the `Wall Duration` field says it took over 100ms:

.. image:: https://user-images.githubusercontent.com/4163446/94200256-1fc17180-fe88-11ea-9935-bef4f818407d.png

Add Paint Method
~~~~~~~~~~~~~~~~

The `MouseMove` event was slow, but why was it slow? Add :meth:`Labels.paint<napari.layer.labels.Label.paint>` to
the trace. Create a new list of callables called `labels` which will trace
the paint method:

.. code-block:: json

    {
        "trace_qt_events": true,
        "trace_file_on_start": "/tmp/latest.json",
        "trace_callables": [
            "labels"
        ],
        "callable_lists": {
            "labels": [
                "napari.layers.labels.Labels.paint"
            ]
        }
    }

Create the new Trace File
~~~~~~~~~~~~~~~~~~~~~~~~~

Run `add_labels` as before, use the paint tool, exit with the **Quit**
command.

View the new Trace File
~~~~~~~~~~~~~~~~~~~~~~~~~

Drop `/tmp/latest.json` into Chrome again. After clicking on the event
press the `m` key to show the duration of that event on the timeline,
below it says the event took 106.597ms:

.. image:: https://user-images.githubusercontent.com/4163446/94201049-66fc3200-fe89-11ea-9720-6a7ff3c7361a.png

We added the timer for :meth:`Labels.paint<napari.layer.labels.Label.paint>` and now we see that `MouseButtonPress`
takes is slow because :meth:`Labels.paint<napari.layer.labels.Label.paint>` is slow. When
investigating a real problem we might have to add quite a few timers. We can save 
the `callable_list` for future use.

Advanced
~~~~~~~~

Create multiple `callable_lists` and toggle them on or off depending on
what you are investigating. The perfmon overhead is low, but tracing only
what you care about will yield the best performance and lead to trace files
that are easier to understand.

Use the :func:`perf_timer<napari.utils.perf.perf_timer>` context object to
time a single block of code, if you don't want to time an entire function
or method.

Use :func:`add_instant_event<napari.utils.perf.add_instant_event>` and
:func:`add_counter_event<napari.utils.perf.add_counter_event>` to annotate
your trace file with additional information beyond just timing events. These
commands should be removed before merging code into master, they are for
temporary use only.
