(nap-4-async-slicing)=

# NAP-4: asynchronous slicing

```{eval-rst}
:Author: Andy Sweet <andrewdsweet@gmail.com>, Jun Xi Ni, Eric Perlman, Kim Pevey
:Created: 2022-06-23
:Status: Draft
:Type: Standards Track
```

## Abstract

Slicing a layer in napari generates a partial view of the layer's data.
The main use of slicing is to define the data that should be rendered
in napari's canvas based on the dimension slider positions.

This project has two major aims.

1. Slice layers asynchronously.
2. Improve the architecture of slicing layers.

We considered addressing these two aims in two separate projects, especially
as (2) is likely a prerequisite for many acceptable implementations of (1).
However, we believe that pursuing a behavioral goal like (1) should help
prevent over-engineering of (2), while simultaneously providing important and
tangible benefits to napari's users.

Ideally this project covers all napari Layer types, though initial progress
may be scoped to image and points layers.


## Motivation and scope

Currently, all slicing in napari is performed synchronously.
For example, if a dimension slider is moved, napari waits to slice each layer
before updating the canvas. When slicing layers is slow, this blocking behavior
makes interacting with data difficult and napari may be reported as non-responsive
by the host operating system.

![The napari viewer displaying a 2D slice of 10 million random 3D points. Dragging the slider changes the 2D slice, but the slider position and canvas updates are slow and make napari non-responsive.](https://i.imgur.com/CSGQbrA.gif)

There are two main reasons why slicing can be slow.

1. Some layer specific slicing operations perform non-trivial calculations (e.g. points).
2. The layer data is read lazily (i.e. it is not in RAM) and latency from the source may be non-negligible (e.g. stored remotely, napari-omero plugin). 

By slicing asynchronously, we can keep napari responsive while allowing for slow
slicing operations. We could also consider optimizing napari to make (1) less of
a problem, but that is outside the scope of this project.


### Slicing architecture

There are a number of existing problems with the technical design of slicing in napari.

- Layers have too much state [^issue-792] [^issue-1353] [^issue-1775].
- The logic is hard to understand and debug [^issue-2156].
- The names of the classes are not self-explanatory [^issue-1574].

Some of these problems and complexity were caused by a previous effort around
asynchronous slicing in an effort to keep it isolated from the core code base.
By contrast, our approach in this project is to redesign slicing in napari to
provide a solid foundation for asychronous slicing and related future work like
multi-canvas and multi-scale slicing.

### Goals

To summarize the scope of this project, we define a few high level goals.
Each goal has many prioritized features where P0 is a must-have, P1 is a should-have,
and P2 is a could-have. Some of these goals may already be achieved by napari in its
current form, but are captured here to prevent any regression caused by this work.


#### 1. Remain responsive when slicing slow data

- P0. When moving a dimension slider, the slider remains responsive so that I can navigate to the desired location.
	- Slider can be moved when data is in the middle of loading.
	- Slider location does not return to position of last loaded slice after it was moved to a different position.
- P0. When the slider is dragged, only slice at some positions so that I don’t wait for unwanted intermediate slicing positions.
	- Once slider is moved, wait before performing slicing operation, and cancel any prior pending slices (i.e. be lazy).
	- If we can reliably infer that slicing will be fast (e.g. data is a numpy array), consider skipping this delay.
- P0. When slicing fails, I am notified so that I can understand what went wrong.
    - May want to limit the number of notifications (e.g. lost network connection for remote data).
- P1. When moving a dimension slider and the slice doesn’t immediately load, I am notified that it is being generated, so that I am aware that my action is being handled.
	- Need a visual cue that a slice is loading.
	- Show visual cue to identify the specific layer(s) that are loading in the case where one layer loads faster than another.


#### 2. Clean up slice state and logic in layers

- P0. Encapsulate the slice input and output state for each layer type, so that I can quickly and clearly identify those.
	- Minimize number of (nested) classes per layer-type (e.g. `ImageSlice`, `ImageSliceData`, `ImageView`, `ImageLoader`).
- P0. Simplify the program flow of slicing, so that developing and debugging against allows for faster implementation. 
	- Reduce the complexity of the call stack associated with slicing a layer.
	- The implementation details for some layer/data types might be complex (e.g. multi-scale image), but the high level logic should be simple.
- P1. Move the slice state off the layer, so that its attributes only represent the whole data.
	- Layer may still have a function to get a slice.
	- May need alternatives to access currently private state, though doesn't necessarily need to be in the Layer (e.g. a plugin with an ND layer, that gets interaction data from 3D visualization , needs some way to get that data back to ND).
- P2. Store multiple slices associated with each layer, so that I can cache previously generated slices.
	- Pick a default cache size that should not strain most machines (e.g. 0-1GB).
	- Make cache size a user defined preference.


#### 3. Measure slicing latencies on representative examples

- P0. Define representative examples that currently cause *desirable* behavior in napari, so that I can check that async slicing does not degrade those.
 	- E.g. 2D slice of a 3D image layer where all data fits in RAM, but not VRAM.
- P0. Define representative examples that currently cause *undesirable* behavior in napari, so that I can check that async slicing improves those.
	- E.g. 2D slice of a 3D points layer where all data fits in RAM, but not VRAM.
	- E.g. 2D slice of a 3D image layer where all data does not on local storage.
- P0. Define slicing benchmarks, so that I can understand if my changes impact overall timing or memory usage.
	- E.g. Do not increase the latency of generating a single slice more than 10%.
	- E.g. Decrease the latency of dealing with 25 slice requests over 1 second by 50%.
- P1. Log specific slicing latencies, so that I can summarize important measurements beyond granular profile timings.
	- Latency logs are local only (i.e. not sent/stored remotely).
	- Add an easy way for users to enable writing these latency measurements.


### Non-goals

To help clarify the scope, we also define some things that were are not explicit goals of this project and give some insight into why they were rejected.

- Make a single slicing operation faster.
	- Useful, but can be done independently of this work.
- Improve slicing functionality.
	- Useful, but can be done independently of this work.
- Toggle the async setting on or off, so that I have control over the way my data loads.
    - May complicate the program flow of slicing.
- When a slice doesn’t immediately load, show a low level of detail version of it, so that I can preview what is upcoming.
	- Requires a low level of detail version to exist.
	- Should be part of a to-be-defined multi-scale project.
- Store multiple slices associated with each layer, so that I can easily implement a multi-canvas mode for napari.
	- Should be part of a to-be-defined multi-canvas project.
	- Solutions for goal (2) should not block this in the future.
- Open/save layers asynchronously.
    - More related to plugin execution.
- Lazily load parts of data based on the canvas' current field of view.
    - An optimization that is dependent on specific data formats.
- Identify and assign dimensions to layers and transforms.
	- Should be part of a to-be-defined dimensions project.
	- Solutions for goal (2) should not block this in the future.
- Thick slices of non-visualized dimensions.
	- Currently being prototyped in [^pull-4334].
	- Solutions for goal (2) should not block this in the future.
- Keep the experimental async fork working.
	- Nice to have, but should not put too much effort into this.

    
## Related work

As this project focuses on re-designing slicing in napari, this section contains information on how slicing in napari currently works.


### Existing slice logic

The following diagram shows the call sequence generated by moving the position of a dimension slider in napari.

![](https://raw.githubusercontent.com/andy-sweet/napari-diagrams/main/napari-slicing-sync-calls.drawio.svg)

Moving the slider generates mouse events that the Qt main event loop handles, which eventually emits napari's `Dims.events.current_step` event, which in turn triggers the refresh of each layer. A refresh first updates the layer's slice state using `Layer.set_view_slice`, then emits the `Layer.events.set_data` event, which finally passes on the layer's new slice state to the vispy scene node using `VispyBaseLayer._on_data_change`.

All these calls occur on the main thread and thus the app does not return to the Qt main event loop until each layer has been sliced and each vispy node has been updated. This means that any other updates to the app, like redrawing the slider position, or interactions with the app, like moving the slider somewhere else, are blocked until slicing is done. This is what causes napari to stop responding when slicing is slow.

Each subclass of `Layer` has its own type-specific implementation of `set_view_slice`, which uses the updated dims/slice state in combination with `Layer.data` to generate and store sliced data. Similarly, each subclass of `VispyBaseLayer` has its own type-specific implementation of `_on_data_change`, which uses the new sliced data in the layer, may post-process it and then passes it to vispy to be rendered on the GPU.

### Existing slice state

It's important to understand what state is currently used by and generated by slicing in napari because solutions for this project may cause this state to be read and write from multiple threads. Rather than exhaustively list all of the slice state of all layer types, we group and highlight some of the more important state. For more context, see the [slicing class and state dependency diagram](https://raw.githubusercontent.com/andy-sweet/napari-diagrams/main/napari-slicing-classes.drawio.svg).

### Input state

Some state that is used as input to slicing is directly mutated by `Layer._slice_dims`. 

- `_ndisplay`: `int`, the display dimensionality (either 2 or 3).
- `_dims_point`: `List[Union[numeric, slice]]`, the current slice position in world coordinates.
- `_dims_order`: `Tuple[int, ...]`, the ordering of dimensions, where the last few are visualized in the canvas.

Note that while `Layer._update_dims` can mutate more state, it should only do so when the dimensionality of the layer has changed, which should not happen when only interacting with existing data.

Other input state comes from more permanent and public properties of `Layer`, which are critical to the slicing operation.

- `data`: array-like, the full data that will be sliced.
- `_transforms`: `TransformChain`, transforms sliced data coordinates to vispy-world coordinates.

Lastly, there are some layer-type specific properties that are needed to show the sliced data in vispy.

- `Points`
    - `face_color`, `edge_color`: `Array[float, (N, 4)]`, the face and edge colors of each point.
    - `size`: `Array[float, (N, D)]`, the size of each point.
    - `edge_width`: `Array[float, (N,)]`, the width of each point's edge.
    - `shown`: `Array[bool, (N,)]`, the visibility of each point.
    - `out_of_slice_display`: `bool`, if True some points may be included in more than one slice based on their size.
- `Shapes`
    - `face_color`, `edge_color`: `Array[float, (N, 4)]`, the face and edge colors of each shape.
    - `edge_width`: `Array[float, (N,)]`, the width of shape's edges.
    - `_data_view`: `ShapeList`, stores all shapes' data.
		- `_mesh`: `Mesh`, stores concatenated meshes of all shapes.
            - `vertices`: `Array[float, (Q, D)]`, the vertices of all shapes.
- `Vectors`
    - `edge_color`: `Array[float, (N,)]`, the color of each vector.
    - `edge_width`: `Array[float, (N,)]`, the width of each vector.
    - `length`: `numeric`, multiplicative length scaling all vectors.
    - `out_of_slice_display`: `bool`, if True some vectors may be included in more than one slice based on their length.
    - `_mesh_vertices`: output from `generate_vector_meshes`.
	- `_mesh_triangles`: output from `generate_vector_meshes`.

These are typically just indexed as part of slicing. For example, `Points.face_color` will be indexed by the points that are visible in the current slice.

#### Output state

The output of slicing is typically layer-type specific, stored as state on the layer, and consumed by the corresponding vispy layer.
    
- `_ImageBase`
    - `_slice`: `ImageSlice`, contains a loader, and the sliced image and thumbnail
        - much complexity encapsulated here and other related classes like `ImageSliceData`.
- `Points`
    - `__indices_view` : `Array[int, (M,)]`, indices of points (i.e. rows of `data`) that are in the current slice.
        - many private properties derived from this (e.g. `_indices_view`, `_view_data`).
    - `_view_size_scale` : `Union[float, Array[float, (M,)]]`, used with thick slices of points `_view_size` to make out of slice points appear smaller.
- `Shapes`
    - `_data_view`: `ShapeList`, stores all shapes' data.
		- `_mesh`: `Mesh`, stores concatenated meshes of all shapes.
			- `displayed_triangles`: `Array[int, (M, 3)]`, triangles to be drawn.
			- `displayed_triangles_colors`: `Array[float, (M, 4)]`, per triangle color.
        according to specified point marker size.
- `Vectors`
    - `_view_indices: Array[int, (-1)]`, indices of vectors that are in the current slice.
        - lots of private properties derived from this (e.g. `_view_face_color`, `_view_faces`).

The vispy layers also read other state from their corresponding layer. In particular they read `Layer._transforms` to produce a transform that can be used to properly represent the layer in the vispy scene coordinate system.

## Detailed description

The following diagram shows the new proposed approach to slicing layers asynchronously.

![](https://raw.githubusercontent.com/andy-sweet/napari-diagrams/main/napari-slicing-async-calls.drawio.svg)

As with the existing synchronous slicing design, the `Dims.events.current_step` event is the shared starting point. In the new approach, we pass `ViewerModel.layers` through to the newly defined `LayerSlicer`, which synchronously (on the main thread) makes a slice request for each layer. This request is processed asynchronously on a dedicated thread for slicing, while the main thread returns quickly to the Qt main event loop, allowing napari to keep responding to other updates and interactions. When all the layers have generated slice responses on the slicing thread, the `slice_ready` event is emitted. That triggers `QtViewer._on_slice_ready` to be executed on the main thread, so that the underlying `QWidgets` can be safely updated.

The rest of this section defines some new types to encapsulate state that is critical
to slicing and some new methods that redefine the core logic of slicing.

### Slice request and response

First, we introduce a type to encapsulate the input to slicing.
A slice request should be immutable and should contain all the state
required to perform slicing.

```python
class _LayerSliceRequest:
    data: ArrayLike
    world_to_data: Transform
    point: Tuple[float, ...]
    dims_displayed: Tuple[int, ...]
    dims_not_displayed: Tuple[int, ...]
```

The expectation is that slicing will capture an immutable instance of this
type on the main thread that another thread can use to perform slicing.
In general, `data` will be too large to copy, so we should instead store a
reference to that, and possibly a weak one to avoid hogging memory unnecessarily.
If `Layer.data` is mutated in-place on the main thread while slicing is being
performed on another thread, this may create an inconsistent slice output
depending on when the values in `data` are accessed, but should be safe.
If `Layer.data` is reassigned on the main thread, then we can safely slice
using all the old data, but we may not want anything to consume the output
because it is now stale.

This definition helps with goal (1) because it allows us to execute asynchronous
slicing without worrying about mutations of layer state on the main thread, which
might be unsafe or create inconsistent output.
It also helps with goal (2) because encapsulating the input to slicing in one type
clarifies exactly what that input is, which is much less clear right now.

Second, we introduce a type to encapsulate the output to slicing.
A slice response should also be immutable and should contain all the state
that consumers need from a slice.

```python
class _LayerSliceResponse:
    data: ArrayLike
    data_to_world: Transform
```

Both these class names include a leading underscore to indicate that they are
private implementation details and external users should not depend on
their existence or any of their fields, as these may change and be
refined over time. This may change in the future, especially for the response
because people may want to handle those in their own way. But there are too
many unknowns to commit to any stability right now.

### Layer methods

We require that each Layer type implements two methods related to slicing.

```python
class Layer:
    ...

    @abstractmethod
    def _make_slice_request(dims: Dims) -> _LayerSliceRequest:
        raise NotImplementedError()

    @abstractmethod
    @staticmethod
    def _get_slice(request: _LayerSliceRequest) -> _LayerSliceResponse:
        raise NotImplementedError()
```

The first, `_make_slice_request`, combines the state of the layer with the
current state of the viewer's instance of `Dims` passed in as a parameter
to create an immutable slice request that slicing will use.
This method should be called from the main thread, so that nothing else
should be mutating the `Layer` or `Dims`.
Therefore, we should expect and try to ensure that this method does not
do too much in order not to block the main thread.

The second, `_get_slice`, takes the slice request and generates a response
using layer-type specific logic.
The method is static to prevent it from using any layer state directly and
instead can only use the state in the slice request. 
This allows us to execute this method on another thread without worrying
about mutations to the layer that might occur on the main thread.

The main consumer of a layer slice response is the corresponding vispy
layer. We require that a vispy layer type implement `_set_slice` to handle
how it consumes the slice output.

```python
class VispyBaseLayer:
    ...

    @abstractmethod
    def _set_slice(self, response: _LayerSliceResponse) -> None:
        raise NotImplementedError()
```

### LayerSlicer object

We define a dedicated class to handle execution of slicing tasks to
avoid the associated state and logic leaking into the already complex
`ViewerModel`.

```python
_ViewerSliceRequest = dict[Layer, _LayerSliceRequest]
_ViewerSliceResponse = dict[Layer, _LayerSliceResponse]

class _LayerSlicer:
    ...

    _executor: Executor = ThreadPoolExecutor(max_workers=1)
    _task: Optional[Future[ViewerSliceResponse]] = None
    ready = Signal(ViewerSliceResponse)

    def slice_layers_async(self, layers: LayerList, dims: Dims) -> None:
        if self._task is not None:
            self._task.cancel()
        requests = {
            layer: layer._make_slice_request(dims)
            for layer in layers
        }
        self._task = self._executor.submit(self._slice_layers, request)
        self._task.add_done_callback(self._on_slice_done)

    def slice_layers(self, requests: ViewerSliceRequest) -> ViewerSliceResponse:
        return {layer: layer._get_slice(request) for layer, request in requests.items()}

    def _on_slice_done(self, task: Future[ViewerSliceResponse]) -> None:
        if task.cancelled():
            return
        self.ready.emit(task.result())
```

While the state and logic is relatively simple right now,
we anticipate that this might grow over time, further motivating a distinct type.

For this class to be useful, there should be at least one connection to the `ready` signal.
In napari, we expect the `QtViewer` to marshall the slice response that this signal carries
to the vispy layers so that the canvas can be updated.

Again this class is marked as private because it's unlikely this
definition will be stable in the short term.
It gives us enough for a minimal version of asynchronous slicing,
but in the future we may want to use more than thread in which case
we may need to add to this definition.


### Hooking up the viewer

Using Python's standard library threads in the `ViewerModel`
mean that we have a portable way to perform asynchronous slicing
in napari without an explicit dependency on Qt.

```python
class ViewerModel:
    ...

    dims: Dims
    _slicer: _LayerSlicer = _LayerSlicer()

    def __init__(self, ...):
        ...
        self.dims.events.current_step.connect(self._slice_layers_async)

    ...

    def _slice_layers_async(self) -> None:
        self._slicer.slice_layers_async(self.layers, self.dims)
```

The main response to the slice being ready occurs on the `QtViewer`.
That's because `QtViewer.layer_to_visual` provides a way to map from
a layer to its corresponding vispy layer.

It's also because `QtViewer` is a `QObject` that lives in the main
thread so can ensure that the slice response is handled on the main
thread. That's useful because consuming the slice response is likely
to update some instances of `QWidget`, which can only be safely updated
on the main thread. Those `QWidgets` may be native to napari or they
may be defined by plugins that respond to napari events.

```python
class QtViewer:
    ...

    viewer: ViewerModel
    layer_to_visual: Dict[Layer, VispyBaseLayer]

    def __init__(self, ...):
        ...
        self.viewer._slicer.ready.connect(self._on_slice_ready)

    @ensure_main_thread
    def _on_slice_ready(self, responses: ViewerSliceResponse):
        for layer, response in responses.items():
            if visual := self.layer_to_visual[layer]:
                visual._set_slice(response)
```

In general, updates to vispy nodes should be done on the main thread [^vispy-faq-threads].
From some prototyping, it seems like Qt backends may be safe
possibly because Qt's signals, slots, and thread affinity can achieve
some automatic thread safety, but that should probably not be relied on.


## Implementation

This section lists the major steps required to implement the NAP. Where
possible, it should be noted where one step is dependent on another, and which
steps may be optionally omitted. Where it makes sense, each step should
include a link to related pull requests as the implementation progresses.

Any pull requests or development branches containing work on this NAP
should be linked to from here. (A NAP does not need to be implemented in a
single pull request if it makes sense to implement it in discrete phases).

If a new NAP document is created, it should be added to the documentation Table
of Contents as an item on `napari/docs/_toc.yml`.


## Backward compatibility

### Breaks synchronous slicing behavior

The main goal of this project is to perform slicing asynchronously, so it's natural that we might break anyone that was depending on slicing being synchronous. At a minimum, we must provide a public way to achieve the same goals. Connecting to the `slice_ready` signal should be sufficient, but that currently contains a privately typed response, so may need consideration.

### Store existing slice state on layer

Many napari behaviors depend on the existing slice input and output state on the layer instances. In this proposal, we decide not to remove this state from the layer yet to prevent breaking other functionality that relies on it. As slice output is generated asynchronously, we must ensure that this state is read and written atomically to mutually exclude the main and slicing thread from reading and/or writing inconsistent parts of that state.

In order to do this, we plan to encapsulate the input and output state of each state into private dataclasses. There are no API changes, but this forces any read/write access of this state to acquire an associated lock.


## Future work

### Render each slice as soon as it is ready

In this proposal, the slicing thread waits for slices of all layers to be ready before it emits the `slice_ready` signal. There are a few reasons for that.

1. We only use one slicing thread to keep behavior simple and to avoid GIL contention.
2. It's closer to the existing behavior of napari
3. Shouldn't introduce any new potential bugs (e.g. (#2862)[https://github.com/napari/napari/issues/2862]).
4. It doesn't need any UX design work to decide what should be shown while we are waiting for slices to be ready.

In some cases, rendering slices as soon as possible will provide a better user experience, especially when some layers are substantially slower than others. Therefore, this should be high priority future work. One way to implement this behavior is to emit a `slice_ready` signal per layer that only contains that layer's slice response.


## Alternatives

- Just call `Layer.set_view_slice` asynchronously, and just leave all existing slice state on `Layer`.
    - Simple to implement and shouldn't break anything that is currently dependent on such state.
    - Needs at least one lock to prevent safe/sensible read/write access to layer slice state (e.g. a lock to control access to the entire layer)
    - How to handle events that should probably be emitted on the main thread?
    - Does not address goal 2.
    
- Only access `Layer.data` asynchronously.
    - Targets main cause of unresponsiveness (i.e. reading data).
    - No events are emitted on the non-main thread.
    - Less lazy when cancelling is possible (i.e. we do more work on the main thread before submitting the async task).
    - Splits up slicing logic into pre/post data reading, making program flow harder to follow.
    - Does not address goal 2.
    
- Use `QThread` and similar utilities instead of `concurrent.futures`
    - Standard way for plugins to support long running operations.
    - Can track progress and allow more opportunity for cancellation with `yielded` signal.
    - Can easily process done callback (which might update Qt widgets) on main thread.
    - Need to define our own task queue to achieve lazy slicing.
    - Need to connect a `QObject`, which ties our core to Qt, unless the code that controls threads does not live in core.
    
- Use `asyncio` package instead of `concurrent.futures`
    - Mostly syntactic sugar on top of `concurrent.futures`.
    - Likely need an `asyncio` event loop distinct from Qt's main event loop, which could be confusing and cause issues.


## Discussion

- [Initial announcement and on Zulip](https://napari.zulipchat.com/#narrow/stream/296574-working-group-architecture/topic/Async.20slicing.20project).
    - Consider (re)sampling instead of slicing as the name for the operation discussed here.  
- [Problems with `NAPARI_ASYNC=1`](https://forum.image.sc/t/even-with-napari-async-1-data-loading-is-blocking-the-ui-thread/68097/4)
- [Removing slice state from layer](https://github.com/napari/napari/issues/4682)
    
### Open questions

- Should we invert design and submit async task with in vispy layer?
    - Is there a way to wait for all layers to be sliced?
        - Maybe if we're slicing via `QtViewer` because that way we could wait for all futures to be done (on another non-main thread) before actually updating the vispy nodes. But that is a little complicated.
        - Probably need to have a design solution for showing slices ASAP to pursue this.
    - I think this design implies that slice state should not live in the model in the future.
        - This might cause issues with selection and other things.
    - Cleans up request/response typing because no need to refer to any base class.
    - Can probably use `@ensure_main_thread` in vispy layer (i.e. for done callback to push data to vispy nodes) because it's private and I think we only intend to support Qt backends for vispy.

- Should we pursue a simpler design first with fewer changes?
    - Slicing accesses pretty much all layer state, so basically need to lock all of that.
        - May introduce long lock contentions, which may cause undesirable behavior.
    - Also need to consider refresh there.

- Can we incrementally implement this design?
    - E.g. one layer type at a time.
    - Yes. We can separate sync layers from async layers in `_LayerSlicer`.
        - This also gives us a long term way to support forcing sync slicing in some cases.

- Should `Dims.current_step` represent the last slice position request or the last slice response?
    - With sync slicing, there is no distinction.
    - If it represents the last slice response, then what should we connect the sliders to?
    - Similar question for `corner_pixels` and others.

## References and footnotes

All NAPs should be declared as dedicated to the public domain with the CC0
license [^cc0], as in `Copyright`, below, with attribution encouraged with
CC0+BY [^cc0-by].

[^cc0]: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication, <https://creativecommons.org/publicdomain/zero/1.0/>

[^cc0-by]: CO0+BY, <https://dancohen.org/2013/11/26/cc0-by/>

[^issue-792]: napari issue 792, <https://github.com/napari/napari/issues/792>

[^issue-1353]: napari issue 1353, <https://github.com/napari/napari/issues/1353>

[^issue-1574]: napari issue 1574, <https://github.com/napari/napari/issues/1574>

[^issue-1775]: napari issue 1775, <https://github.com/napari/napari/issues/1775>

[^issue-2156]: napari issue 2156, <https://github.com/napari/napari/issues/2156>

[^pull-4334]: napari pull request 4334, <https://github.com/napari/napari/pull/4334>

[^vispy-faq-threads]: Vispy FAQs: Is VisPy multi-threaded or thread-safe?, <https://vispy.org/faq.html#is-vispy-multi-threaded-or-thread-safe>

## Copyright

This document is dedicated to the public domain with the Creative Commons CC0
license [^cc0]. Attribution to this source is encouraged where appropriate, as per
CC0+BY [^cc0-by].
