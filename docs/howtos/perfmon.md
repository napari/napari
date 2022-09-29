(perfmon)=

# Performance monitoring

If napari is not performing well, you can use
{mod}`napari.utils.perf<napari.utils.perf>` to help
diagnose the problem.

The module can do several things:

1. Time Qt Events

2. Display a dockable **performance** widget.

3. Write JSON trace files viewable with `chrome://tracing`.

4. Time any function that you specify in the config file.

## Monitoring vs. profiling

Profiling is similar to performance monitoring. However profiling usually
involves running an external tool to acquire timing data on every function
in the program. Sometimes this will cause the program to run so slowly it's
hard to use the program interactively.

Performance monitoring does not require running a separate tool to collect
the timing information, however we do use Chrome to view the trace files.
With performance monitoring napari can run at close to full speed in many
cases. This document discusses only napari's performance monitoring
features. Profiling napari might be useful as well, but it is not discussed
here.


## Enabling perfmon

There are two ways to enable performance monitoring. Set the environment
variable `NAPARI_PERFMON=1` or set `NAPARI_PERFMON` to the path of 
a JSON configuration file, for example `NAPARI_PERFMON=/tmp/perfmon.json`.

```{note}
Note: when using `NAPARI_PERFMON`, napari must create the Qt Application.
If you are using `NAPARI_PERFMON=1 ipython`, do not use `%gui qt` before
creating a napari `Viewer`.
```

Setting `NAPARI_PERFMON=1` does three things:

1. Times Qt Events
2. Shows the dockable **performance** widget.
3. Reveals the **Debug** menu which you can use to create a trace file.

## Configuration file format

Example configuration file:

```json
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
```

## Configuration options

### `trace_qt_events`

If true perfmon will time the duration of all Qt Events. You might
want to turn this off if the overhead is noticeable, or if you want
your trace file to be less cluttered.

### `trace_file_on_start`

If a path is given, napari will start tracing immediately on start. In many
cases this is much more convenient than using the **Debug** Menu. Be sure to
exit napari using the **Quit** command. The trace file will be written on
exit.

### `trace_callables`

Specify which `callable_lists` you want to trace. You can have many
`callable_lists` defined, but this setting says which should be traced.

### `callable_lists`

These lists can be referenced by the `callable_lists` option. You might
want multiple lists so they can be enabled separately.

## Trace file

The trace file that napari produces is viewable in Chrome. Go to the
special URL `chrome://tracing`. Use the **Load** button inside the Chrome
window, or just drag-n-drop your JSON trace file into the Chrome window.
You can also view trace files using the
[Speedscope website](https://www.speedscope.app/).
It is similar to `chrome://tracing` but has some different features.

The trace file format is specified in the
[Trace File Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview)
Google Doc. The format is well-documented, but there are no pictures so
it's not always clear how a given feature actually looks in the Chrome
Tracing GUI.

## Example investigation

This is an example showing how you might use the
{mod}`napari.utils.perf<napari.utils.perf>` module.

### Add a sleep

To simulate a performance problem in napari, add a `sleep()` call to the
{meth}`Labels.paint<napari.layer.labels.Label.paint>` method, this
will make the method take at least 100 ms:

```{code-block} python
:emphasize-lines: 2-3

def paint(self, coord, new_label, refresh=True):
    import time
    time.sleep(0.1)

    if refresh is True:
        self._save_history()
```

### Create a perfmon config file

Create a minimal perfmon config file `/tmp/perfmon.json` that looks like this:

```json
{
    "trace_qt_events": true,
    "trace_file_on_start": "/tmp/latest.json",
    "trace_callables": []
}
```

This will write `/tmp/latest.json` every time we run napari. This file is
only written on exit, and you must exit with the **Quit** commmand. Using
`trace_file_on_start` is often easier than manually starting a trace using
the **Debug** menu.

### Run napari

Now run napari's `add_labels` example like this:

```shell
    NAPARI_PERFMON=/tmp/perfmon.json python examples/add_labels.py
```

Use the paint tool and single-click once or twice on the labels layer. Look
at the **performance** widget, it should show that some events took over
100ms. The **performance** widget is just to give you a quick idea of what
is running slow:

![example widget output ](https://user-images.githubusercontent.com/4163446/94198620-898c4c00-fe85-11ea-8769-83f52c0a1aad.png)

The trace file will give you much more information than the **performance**
widget. Exit napari using the **Quit** command so that it writes the trace
file on exit.

### View trace in Chrome

Run Chrome and go to the URL `chrome://tracing`. Drag and drop
`/tmp/latest.json` into the Chrome window, or use the **Load** button to
load the JSON file. You will usually need to pan and zoom the trace to
explore it, to figure out what is going on.

You can navigate with the mouse, but using the keyboard might be easier.
Press the `AD` keys to move left and right, and press the `WS` keys to zoom
in or out. Both the `MouseButtonPress` and `MouseMove` events are slow. In
the lower pane the `Wall Duration` field says it took over 100ms:

![example chrome output ](https://user-images.githubusercontent.com/4163446/94200256-1fc17180-fe88-11ea-9935-bef4f818407d.png)

So we can see that some events are running slow. The next questions is
why are `MouseButtonPress` and `MouseMove` running slow? To answer this
question we can add more timers. In this case we know the answer, but often
you will have to guess or experiment. You might add some timers and then
find out they actually run fast, so you can remove them.

### Add paint method

To add the {meth}`Labels.paint<napari.layers.Labels.paint>` method to
the trace, create a new list of callables named `labels` and put the
{meth}`Labels.paint<napari.layers.Labels.paint>` method into 
that list.

```json
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
```

### Create the new trace File

Run `add_labels` as before, click with the paint tool, exit with the **Quit**
command.

### View the new trace File

Drop `/tmp/latest.json` into Chrome again. Now we can see that
`MouseButtonPress` calls
{meth}`Labels.paint<napari.layers.Labels.paint>` and that
{meth}`Labels.paint<napari.layers.Labels.paint>` is really responsible
for most of the time. After clicking on the event press the `m` key, that
will highlight the event duration with arrows and print the duration right
on the timeline, in this case it says the event took 106.597ms:

![highlighted event duration showing Labels.paint took 106.597ms to run ](https://user-images.githubusercontent.com/4163446/94201049-66fc3200-fe89-11ea-9720-6a7ff3c7361a.png)

When investigating a real problem we might have to add many functions to
the config file. It's best to add timers that take a lot of time. If you
add a timer that's called thousands of times, it will add overhead and will
clutter the trace file. In general we want to trace important and
interesting functions. If we create a large `callable_list` we can save it
for future use.

### Advanced

Experiment with the {mod}`napari.utils.perf<napari.utils.perf>` features and
you will find your own tricks and techniques.

Create multiple `callable_lists` and toggle them on or off depending on
what you are investigating. The perfmon overhead is low, but tracing only
what you care about will yield the best performance and lead to trace files
that are easier to understand.

Use the {func}`perf_timer<napari.utils.perf.perf_timer>` context object to
time only a block of code, or even a single line, if you don't want to time
an entire function.

Use {func}`add_instant_event<napari.utils.perf.add_instant_event>` and
{func}`add_counter_event<napari.utils.perf.add_counter_event>` to annotate
your trace file with additional information beyond just timing events. The
`add_instant_event` function draws a vertical line on the trace in Chrome,
to show when something happened like a click. The `add_counter_event`
function creates a bar graph on the trace showing the value of some counter
at every point in time. For example you could record the length of a queue,
and see the queue grow and shrink over time.

Calls to `perf_timer`, `add_instant_event` and `add_counter_event` should
be removed before merging code into main. Think of them like "debug
prints", things you add while investigating a problem, but you do not leave
them in the code permanently.

You can save JSON files so that you can compare how things looked
before and after your changes.
