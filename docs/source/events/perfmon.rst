.. _perfmon:

Performance Monitoring
======================

Performance is a core feature of napari. In order to help you monitor
performance and diagnose or fix performance issues, napari includes
:mod:`~napari.utils.perf`. The module has several features:

1. It can time Qt Events and other functions or methods.

2. It can display a dockable performance widget that will list events whose
duration is over some configurable threshold.

3. It can produce JSON trace files that can be visualized and interactively
explored using Chrome's tracing GUI.

Enabling perfmon
----------------

To enable perfmonance monitoring set the environment variable
`NAPARI_PERFMON=1`, or is set to the path of a configuration file such as
`NAPARI_PERFMON=~/.perfmon.json`.

Setting the environment variable to 1 enables perfmon in a mode that can
only time Qt Events. In order to time other functions and methods, you need
to use the configuration file.

Configuration File
------------------

.. code-block:: JSON
    {
        "trace_qt_events": true,
        "trace_file_on_start": "/path/to/latest.json",
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

Config options
--------------

`trace_qt_events` if true perfmon times the duration of all Qt Events. This is
on by default if `NAPARI_PERFMON=1`. The only reason to turn it off would be
if the overhead is noticeable, or if you just wanted less clutter in your trace files.

`trace_file_on_start` if set then napari will start recording a trace file
as soon as it starts. In many cases this is much more convenient than using
the Debug Menu. Simply run napari, do what you want to test, and exit. Be
sure to exist with the Quit option. The trace file is only written on exit,
because otherwise writing the file would alter the timing of things.

`trace_callables` is how you trace function and methods. Tracing Qt Events
is a great start, but often you will need to add many functions and methods
during the investigation of a performance issue.

`callable_lists` let you define lists of callables that you can toggle on
with the `trace_callables` setting. The idea is you can have a number of
pre-defined sets of callables, and then toggle on just the ones you are
interested in. The reason to not just trace everything is tracing has some
performance overhead of its own, and tracing too many things will clutter
the trace file.

Trace File
-----------

A performance trace is a JSON file that is viewable using Chrome's tracing
GUI. The GUI is a development tool that's built-in to Chrome. You can
access the GUI by going to the special URL `chrome://tracing` in Chrome.
You can also those same files using the Speedscope website at
https://www.speedscope.app/.

Google "Trace Event Format" for the Google Doc which specifies the trace
file format. The format is well-documented, but the document does not
generally explain how a given feature actually looks in the Chrome Tracing
GUI.




