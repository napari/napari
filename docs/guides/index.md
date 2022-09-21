# In-depth explanations

```{note}
These pages describe advanced usage and how napari works internally. If you are
just getting started, check out our [tutorials](../tutorials/index.md) or
[how-to guides](../howtos/index.md) instead.
```

## Advanced usage

magicgui is a python package that assists in building small, composable
graphical user interfaces (widgets). To learn about using `magicgui` in napari,
see the {ref}`magicgui` guide.

If you'd like to start customizing the behavior of napari, it pays to
familiarize yourself with the concept of an Event Loop. For an introduction to
event loops and connecting your own functions to events in napari, see the
{ref}`intro-to-event-loop`.

If you use napari to view and interact with the results of long-running
computations, and would like to avoid having the viewer become unresponsive
while you wait for a computation to finish, you may benefit from reading about
{ref}`multithreading-in-napari`.

If you are interested in using napari to explore 3D objects, see {ref}`3d-interactivity`.

See {ref}`rendering-explanation` for two experimental features than can
optionally be enabled to add non-blocking rendering to napari. You can also
check out the {ref}`dedicated guide on asynchronous rendering in napari <rendering>`.

To understand how to test and measure performance in napari, see {ref}`napari-performance`.

## Architecture documents

See {ref}`napari-preferences` for the list of preferences that can be set via
the preferences dialog.

If you are writing a plugin manifest, or are interested in contributing to
napari, you can also read about the concept of {ref}`context-expressions`.

For a full list of events you may connect to, see {ref}`events-reference`.
