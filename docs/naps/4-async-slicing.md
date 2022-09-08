(nap-4-async-slicing)=

# NAP-4: asynchronous slicing

```{eval-rst}
:Author: Andy Sweet <andrewdsweet@gmail.com>, Jun Xi Ni, Eric Perlman, Kim Pevey
:Created: 2022-06-23
:Resolution: TODO <url>
:Resolved: TODO <yyyy-mm-dd>
:Status: Accepted
:Type: Standards Track
:Version effective: 0.5
```

## Abstract

Slicing a layer in napari generates a partial view of the layer's data based
on the dimension slider positions and the current field of view.

This project has two major goals.

1. Slice layers asynchronously to keep napari responsive while allowing for slow slicing.
2. Improve the logic of slicing layers to help developers better understand how it works and unlock new feature development.

We propose a lock-free asynchronous solution that uses Python's built-in
`concurrent.futures` module and exploits napari's main event loop.
For each slice, an immutable shallow copy of all the required input to slicing is
collected on the main thread and is then processed on a separate single-threaded executor.
When the slice task is finished, we notify vispy and Qt objects back on the main thread
so they can be safely updated.
This approach allows us to gradually introduce asynchronous slicing to napari
layer by layer without breaking other functionality that relies on existing layer slice state.


## Motivation and scope

Currently, all slicing in napari is performed synchronously.
For example, if a dimension slider is moved, napari waits to slice a layer
before updating the canvas. When slicing layers is slow, this blocking behavior
makes interacting with data difficult and napari may be reported as not responding
by the host operating system.

![The napari viewer displaying a 2D slice of a 3D multi-resolution electron microscopy image stored remotely. Dragging the slider changes the 2D slice, but the slider position and canvas updates are slow and napari stops responding.](https://i.imgur.com/cAJxkLq.gif)

There are two main reasons why slicing can be slow.

1. Some layer specific slicing operations perform non-trivial calculations (e.g. points).
2. The layer data is read lazily (i.e. it is not in RAM) and latency from the source may be non-negligible (e.g. stored remotely, napari-omero plugin). 

By slicing asynchronously, we can keep napari responsive while allowing for slow
slicing operations. We could also consider optimizing napari to make (1) less of
a problem, but that is outside the scope of this project.

Introducing asynchrony to software often complicates program flow and logic.
However, the existing technical design of slicing is already complicated and
often causes issues for developers.

- Layers have too much state [^issue-792] [^issue-1353] [^issue-1775].
- The logic is hard to understand and debug [^issue-2156].
- The names of the classes are not self-explanatory [^issue-1574].

Some of these issues were caused by a previous effort around asynchronous slicing
in an effort to keep it isolated from the core code base.
By contrast, our approach in this project is to redesign slicing in napari to
provide a solid foundation for asychronous slicing and related future work like
multi-canvas and multi-scale slicing.

### Goals

To summarize the scope of this project, we define a few high level goals.
Each goal has many prioritized features where P0 is a must-have, P1 is a should-have,
and P2 is a could-have. Some of these goals may already be achieved by napari in its
current form, but are captured here to prevent any regression caused by this work.


#### 1. Keep responding when slicing slow data

- P0. When moving a dimension slider, the slider can still be controlled so that I can navigate to the desired location.
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
	- E.g. 2D slice of a 3D image layer where all data is not on local storage.
- P0. Define slicing benchmarks, so that I can understand if my changes impact overall timing or memory usage.
	- E.g. Do not increase the latency of generating a single slice more than 10%.
	- E.g. Decrease the latency of dealing with 25 slice requests over 1 second by 50%.
- P1. Log specific slicing latencies, so that I can summarize important measurements beyond granular profile timings.
	- Latency logs are local only (i.e. not sent/stored remotely).
	- Add an easy way for users to enable writing these latency measurements.


### Non-goals

To help clarify the scope, we also define some things that were are not explicit goals of this project and briefly explain why they were omitted.

- Make a single slicing operation faster.
	- Useful, but can be done independently of this work.
- Improve slicing functionality.
	- Useful, but can be done independently of this work.
- Make async a toggleable setting (i.e. async is always on).
    - May complicate the program flow of slicing.
    - May still automatically infer that slicing can be synchronously.
- When a slice doesn’t immediately load, show a low level of detail version of it, so that I can preview what is upcoming.
	- Requires a low level of detail version to exist.
	- Should be part of a to-be-defined multi-scale project.
- Store multiple slices associated with each layer, so that I can easily implement a multi-canvas mode for napari.
	- Should be part of a to-be-defined multi-canvas project.
	- Solutions for goal (2) should not block this in the future.
- Open, save, or process layers asynchronously.
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
    - May want to remove it to avoid confusion.

    
## Related work

As this project focuses on re-designing slicing in napari, this section contains information on how slicing in napari currently works.


### Existing slice logic

The following diagram shows the simplified call sequence generated by moving the position of a dimension slider in napari.

![](https://raw.githubusercontent.com/andy-sweet/napari-diagrams/main/napari-slicing-sync-calls.drawio.svg)

Moving the slider generates mouse events that the Qt main event loop handles,
which eventually emits napari's `Dims.events.current_step` event, which in turn
triggers the refresh of each layer. A refresh first updates the layer's slice state
using `Layer.set_view_slice`, then emits the `Layer.events.set_data` event,
which finally passes on the layer's new slice state to the vispy scene node
using `VispyBaseLayer._on_data_change`.

All these calls occur on the main thread so the app does not return to the Qt main event
loop until each layer has been sliced and each vispy node has been updated. This means that
any other updates to the app, like redrawing the slider position, or other user interactions
with the app, like moving the slider somewhere else, are blocked until slicing is done.
This is what causes napari to stop responding when slicing is slow.

Each subclass of `Layer` has its own type-specific implementation of `set_view_slice`,
which uses the updated dims state in combination with `Layer.data` to generate and store sliced data.
Similarly, each subclass of `VispyBaseLayer` has its own type-specific implementation of `_on_data_change`,
which gets the new sliced data in the layer, may post-process it, and finally passes it to vispy to be rendered on the GPU.

### Existing slice state

It's important to understand what state is currently used by and generated by slicing because
solutions for this project may cause this state to be read and write from multiple threads.
Rather than exhaustively list all of the slice state of all layer types and their dependencies [^slice-class-diagram],
we group and highlight some of the more important state.

### Input state

Some input state for slicing is directly mutated by `Layer._slice_dims`. 

- `_ndisplay: int`, the display dimensionality (either 2 or 3).
- `_dims_point: List[Union[numeric, slice]]`, the current slice position in world coordinates.
- `_dims_order: Tuple[int, ...]`, the ordering of dimensions, where the last few are visualized in the canvas.

Note that while `Layer._update_dims` can mutate more state, it should only do so when
the dimensionality of the layer has changed, which should not happen when only interacting with existing data.

Other input state comes from more permanent and public properties of `Layer`, which are critical to the slicing operation.

- `data: ArrayLike`, the full data to be sliced.
- `_transforms: TransformChain`, transforms sliced data coordinates to vispy-world coordinates.

Finally, there are some layer-type specific properties that are used to customize the visualized slice.

- `_ImageBase`
    - `rgb: bool` True if the leading dimension contains RGB(A) channels that should be visualized as such.
- `Points`
    - `face_color, edge_color: Array[float, (N, 4)]`, the face and edge colors of each point.
    - `size: Array[float, (N, D)]`, the size of each point.
    - `edge_width: Array[float, (N,)]`, the width of each point's edge.
    - `shown: Array[bool, (N,)]`, the visibility of each point.
    - `out_of_slice_display: bool`, if True some points may be included in more than one slice based on their size.
- `Shapes`
    - `face_color, edge_color: Array[float, (N, 4)]`, the face and edge colors of each shape.
    - `edge_width: Array[float, (N,)]`, the width of shape's edges.
    - `_data_view: ShapeList`, stores all shapes' data.
		- `_mesh: Mesh`, stores concatenated meshes of all shapes.
            - `vertices: Array[float, (Q, D)]`, the vertices of all shapes.
- `Vectors`
    - `edge_color: Array[float, (N,)]`, the color of each vector.
    - `edge_width: Array[float, (N,)]`, the width of each vector.
    - `length: numeric`, multiplicative length scaling all vectors.
    - `out_of_slice_display: bool`, if True some vectors may be included in more than one slice based on their length.
    - `_mesh_vertices`, output from `generate_vector_meshes`.
	- `_mesh_triangles`, output from `generate_vector_meshes`.

Many of these are just indexed as part of slicing.
For example, `Points.face_color` is indexed by the points that are visible in the current slice.

#### Output state

The output of slicing is typically layer-type specific, stored as state
on the layer, and mostly consumed by the corresponding vispy layer.
    
- `_ImageBase`
    - `_slice: ImageSlice`, contains a loader, and the sliced image and thumbnail
        - much complexity encapsulated here and other related classes like `ImageSliceData`.
- `Points`
    - `__indices_view: Array[int, (M,)]`, indices of points (i.e. rows of `data`) that are in the current slice.
        - many private properties derived from this (e.g. `_indices_view`, `_view_data`).
    - `_view_size_scale: Union[float, Array[float, (M,)]]`, used with thick slices of points `_view_size` to make out of slice points appear smaller.
- `Shapes`
    - `_data_view: ShapeList`, stores all shapes' data.
		- `_mesh: Mesh`, stores concatenated meshes of all shapes.
			- `displayed_triangles: Array[int, (M, 3)]`, triangles to be drawn.
			- `displayed_triangles_colors: Array[float, (M, 4)]`, per triangle color.
- `Vectors`
    - `_view_indices: Array[int, (-1)]`, indices of vectors that are in the current slice.
        - lots of private properties derived from this (e.g. `_view_face_color`, `_view_faces`).

The vispy layers also read other state from their corresponding layer.
In particular they read `Layer._transforms` to produce a transform that can be used to properly
represent the layer in the vispy scene coordinate system.

## Detailed description

The following diagram shows the new proposed approach to slicing layers asynchronously.

![](https://raw.githubusercontent.com/andy-sweet/napari-diagrams/main/napari-slicing-async-calls.drawio.svg)

As with the existing synchronous slicing design, the `Dims.events.current_step` event
is the shared starting point. In the new approach, we pass `ViewerModel.layers` through
to the newly defined `LayerSlicer`, which makes a slice request for each layer on the main thread.
This request is processed asynchronously on a dedicated thread for slicing, while the main thread
returns quickly to the Qt main event loop, allowing napari to keep responding to other updates
and interactions. When all the layers have generated slice responses on the slicing thread,
the `slice_ready` event is emitted. That triggers `QtViewer.on_slice_ready` to be executed on
the main thread, so that the underlying `QWidgets` can be safely updated.

The rest of this section defines some new types to encapsulate state that is critical
to slicing and some new methods that redefine the core logic of slicing.

### Request and response

First, we introduce a request to encapsulate the required input to slicing.

```python
class LayerSliceRequest:
    data: ArrayLike
    world_to_data: Transform
    point: Tuple[float, ...]
    dims_displayed: Tuple[int, ...]
    dims_not_displayed: Tuple[int, ...]
```

At a minimum, this request must contain the data to be sliced, the point at
which we are slicing in napari's shared world coordinate system, and a way to
transform world coordinates to a layer's data coordinate system.
In the future, the point may become a bounding box that includes the current
field of view.
Given our understanding about the existing slice input state, each layer type
should extend this request type to capture the input it needs for slicing, such
as `Points.face_color`.

Second, we introduce a response to encapsulate the output of slicing.

```python
class LayerSliceResponse:
    data: ArrayLike
    data_to_world: Transform
```

At a minimum, this response must contain the sliced data and a transform that
tells us where that data lives in the currently displayed scene of napari's
shared world coordinate system.
At this point `data` should be fast to access either from RAM (e.g. as a numpy
array) or VRAM (e.g. as a CuPy array), so that it can be consumed on the main
thread without blocking other operations for long.

For each layer type, we need to extend the request and response types to include
state that is specific to slicing those types of layers. For example, the image
request type should look something like the following.

```python
class ImageSliceRequest(LayerSliceRequest):
    rgb: bool
    multiscale: bool
    corner_pixels: np.ndarray
    data_level: int

class ImageSliceResponse(LayerSliceResponse)
    thumbnail: np.ndarray
```

As napari supports RGB and multi-scale images, we need a few more inputs to
know what data should be accessed to generate the desired slice for the canvas.

The points request and response types should look something like the following.

```python
class PointsSliceRequest(LayerSliceRequest):
    out_of_slice_display: bool
    size: np.ndarray
    face_color: np.ndarray
    edge_color: np.ndarray
    edge_width: np.ndarray
    shown: np.ndarray

class PointsSliceResponse(LayerSliceResponse):
    indices: np.ndarray
    size: np.ndarray
    view_size_scale: Union[int, np.ndarray]
    face_color: np.ndarray
    edge_color: np.ndarray
    edge_width: np.ndarray
```

While the names and typing of some of the points request and response attributes
are the same (e.g. `face_color`) they differ semantically because the response
represent the sliced values only. Defining two distinct and explicit request and
response types gives us more flexibility later in terms of how slicing works.
For example, we may want to generate point face colors lazily, which could mean
that `PointsSliceRequest.face_color` would become a callable instead of a
materialized array as in the response.


### Layer methods

We require that each Layer type implements two methods related to slicing.

```python
class Layer:
    ...

    @abstractmethod
    def _make_slice_request(dims: Dims) -> LayerSliceRequest:
        raise NotImplementedError()

    @abstractmethod
    @staticmethod
    def _get_slice(request: LayerSliceRequest) -> LayerSliceResponse:
        raise NotImplementedError()
```

The first, `_make_slice_request`, combines the state of the layer with the
current state of the viewer's instance of `Dims` passed in as a parameter
to create an immutable slice request that the slicing operation will use
on another thread.
This method should be called from the main thread, so that nothing else
should be mutating the `Layer` or `Dims`.
Therefore, we should try to ensure that this method returns quickly, so as 
not to block the main thread.

Most of the request's fields, like `point` and `dims_displayed` are small,
and can be quickly copied in this new instance.
Other fields, like `data`, are too large to copy in general, so instead we
store a reference.
If that reference is mutated in-place on the main thread while slicing is being
performed on another thread, this may create an inconsistent slice output
depending on when the values in `data` are accessed, but should be safe because
the `shape` and `dtype` cannot change.
If `Layer.data` is reassigned on the main thread, then we can safely slice
using the reference to the old data, though we may not want to consume
the now stale output.

The second, `_get_slice`, takes the slice request and generates a response
using layer-type specific logic.
The method is static to prevent it from using any layer state directly and
instead can only use the state in the immutable slice request. 
This allows us to execute this method on another thread without worrying
about mutations to the layer that might occur on the main thread.

The main consumer of a layer slice response is the corresponding vispy
layer. We require that a vispy layer type implement `_set_slice` to handle
how it consumes the slice output.

```python
class VispyBaseLayer:
    ...

    @abstractmethod
    def _set_slice(self, response: LayerSliceResponse) -> None:
        raise NotImplementedError()
```

### Layer slicer

We define a dedicated class to handle execution of slicing tasks to
avoid the associated state and logic leaking into the already complex
`ViewerModel`.

```python
ViewerSliceRequest = dict[Layer, LayerSliceRequest]
ViewerSliceResponse = dict[Layer, LayerSliceResponse]

class LayerSlicer:
    ...

    _executor: Executor = ThreadPoolExecutor(max_workers=1)
    _task: Optional[Future[ViewerSliceResponse]] = None

    def __init__(self, ...):
        self.events = EmitterGroup(source=self, slice_ready=Event)

    def slice_layers_async(self, layers: LayerList, dims: Dims) -> Future[ViewerSliceResponse]:
        if self._task is not None:
            self._task.cancel()
        requests = {
            layer: layer._make_slice_request(dims)
            for layer in layers
        }
        self._task = self._executor.submit(self._slice_layers, request)
        self._task.add_done_callback(self._on_slice_done)
        return self._task

    def slice_layers(self, requests: ViewerSliceRequest) -> ViewerSliceResponse:
        return {layer: layer._get_slice(request) for layer, request in requests.items()}

    def _on_slice_done(self, task: Future[ViewerSliceResponse]) -> None:
        if task.cancelled():
            return
        self.events.slice_ready(task.result())
```

For this class to be useful, there should be at least one connection to the `slice_ready` event.
In napari, we expect the `QtViewer` to marshall the slice response that this signal carries
to the vispy layers so that the canvas can be updated.

### Hooking up the viewer

Using Python's standard library threads in the `ViewerModel`
means that we have a portable way to perform asynchronous slicing
in napari without an explicit dependency on Qt.

```python
class ViewerModel:
    ...

    dims: Dims
    _slicer: LayerSlicer = LayerSlicer()

    def __init__(self, ...):
        ...
        self.dims.events.current_step.connect(self._slice_layers_async)

    ...

    def _slice_layers_async(self) -> None:
        self._slicer.slice_layers_async(self.layers, self.dims)
```

The main response to the slice being ready occurs on the `QtViewer`
because `QtViewer.layer_to_visual` provides a way to map from
a layer to its corresponding vispy layer.

```python
class QtViewer:
    ...

    viewer: ViewerModel
    layer_to_visual: Dict[Layer, VispyBaseLayer]

    def __init__(self, ...):
        ...
        self.viewer._slicer.events.slice_ready.connect(self._on_slice_ready)

    @ensure_main_thread
    def _on_slice_ready(self, responses: ViewerSliceResponse):
        for layer, response in responses.items():
            if visual := self.layer_to_visual[layer]:
                visual._set_slice(response)
```

The other reason is because `QtViewer` is a `QObject` that lives in the main
thread, which makes it easier to ensure that the slice response is handled
on the main thread.
That's useful because Qt widgets and vispy nodes can only be safely updated
on the main thread [^vispy-faq-threads], both of which occur when consuming
slice output.


## Implementation

A tracking issue [^tracking-issue] serves as a way to communicate high level
progress and discussion.

A prototype [^prototype-pr] has been created that implements the approach described
above to show its basic technical feasibility in napari's existing code base.
It only implements it for some layer types, some slicing operations, and may
not work when layer data is mutated.
There are some other rough edges and plenty of ways to break it, but basic
functionality should work.

The rough implementation plan is as follows.

- Implement pure refactors related to slicing to support this work (no public API or behavior changes).
- Implement high-level asynchronous slicing model described here (i.e. add `LayerSlicer`, changes to `ViewerModel`), but with existing layer slicing implementation (i.e. `Layer._set_view_slice`).
- Incrementally add layer slicing implementation described here (i.e. `Layer._get_slice`) for each layer separately.

This allows us to merge small pieces of work into `main`, testing and fixing
them as we go. The alternative is to maintain a large branch/fork of napari that
may be less tested and harder to merge in the end.

## Backward compatibility

### Breaks synchronous slicing behavior

The main goal of this project is to perform slicing asynchronously, so it's
natural that we might break anyone that was depending on slicing being synchronous.
In particular, users or plugins that rely on `Viewer.screenshot` to capture the
contents of the canvas after updating things like `Dims.current_step` will be affected.

Our proposal will break that usage in general, but we plan to offer some public way to
force synchronous slicing. For example, one promising idea is to define a context manager
that would temporarily force `LayerSlicer` to always wait for asynchronous tasks to finish
and could be used as follows.

```python
with viewer.sync_slicing():
    viewer.dims.current_step = (10, 0, 0)
viewer.screenshot(...)
```

Having such a mechanism in `LayerSlicer` also allows us to implement support for
async slicing incrementally for each layer type.

### Store existing slice state on layer

Many existing napari behaviors depend on the existing slice input and output state
on the layer instances. In this proposal, we decide not to remove this state from the
layer yet to prevent breaking other functionality that relies on it. As slice output is
now generated asynchronously, we must ensure that this state is read and written atomically
to mutually exclude the main and slicing thread from reading and writing inconsistent
parts of that state.

In order to do this, we plan to encapsulate the input and output state of each state into
private dataclasses. There are no API changes, but this forces any read/write access of
this state to acquire an associated lock.


## Future work

### Render each slice as soon as it is ready

In this proposal, the slicing thread waits for slices of all layers to be ready before
it emits the `slice_ready` event. There are a few reasons for that.

1. We only use one slicing thread to keep behavior simple and to avoid GIL contention.
2. It's closer to the existing behavior of napari
3. Shouldn't introduce any new potential bugs, such as [^issue-2862].
4. It doesn't need any UX design work to decide what should be shown while we are waiting for slices to be ready.

In some cases, rendering slices as soon as possible will provide a better user experience,
especially when some layers are substantially slower than others. Therefore, this should be
high priority future work. One way to implement this behavior is to emit a `slice_ready`
event per layer that only contains that layer's slice response.


## Alternatives

### Extend the existing experimental async code

- Already being used somewhat successfully by some napari users in the wild.
- Can still reuse some of the code, tooling, and learnings from this project.
- The technical design of this approach was not well received (i.e. there is a reason it is experimental).
- Does not address goal 2.

### Just call `set_view_slice` or `refresh` asynchronously

- Simple to implement with few code changes needed.
- Needs at least one lock to provide sensible access to layer slice state.
- This will need to be acquired to access this state and at the beginning of many methods that access any of that state.
- How to emit events on the main thread?
- Does not address goal 2.
    
### Just access `data` asynchronously

- Targets main cause of unresponsiveness (i.e. reading data).
- No events are emitted on the non-main thread.
- Less lazy when cancelling is possible (i.e. we do more work on the main thread before submitting the async task).
- Splits up slicing logic into pre/post data reading, making program flow harder to follow.
- Does not address goal 2.
    
### Use `QThread` and similar utilities instead of `concurrent.futures`

- Standard way for plugins to support long running operations.
- Can track progress and allow more opportunity for cancellation with `yielded` signal.
- Can easily process done callback (which might update Qt widgets) on main thread.
- Need to define our own task queue to achieve lazy slicing.
- Need to connect a `QObject`, which ties our core to Qt, unless the code that controls threads does not live in core.
    
### Use `asyncio` package instead of `concurrent.futures`

- May improve general readability of async code for some.
- Mostly syntactic sugar on top of `concurrent.futures`.
- Likely need an `asyncio` event loop distinct from Qt's main event loop, which could be confusing and cause issues.
- [As discussed on Zulip](https://napari.zulipchat.com/#narrow/stream/212875-general/topic/qt-async-threads), `qt-async-threads` is a possible solution,
  but it is still quite new.

### Slice from vispy layer instead of viewer model

Instead of making the `ViewerModel` the driver of slicing, we could instead drive it from the vispy layers.
A rough implementation of this approach could look like the following.

```python

class QtViewer:
    ...
    def __init__(self):
        ...
        self.viewer.dims.current_step.connect(self.slice_layers)

    def slice_layers(self):
        for layer in self.viewer.layers:
            self.layer_to_visual[layer]._slice(self.viewer.dims)

class Image:
    ...
    def _make_slice_request(self, dims: Dims) -> ImageSliceRequest:
        ...
    def _get_slice(self, request: ImageSliceRequest) -> ImageSliceResponse:
        ...

class VispyImageLayer:
    ...
    def _slice(self, dims: Dims) -> None:
        request = self.layer._make_slice_request(dims)
        task = self.slice_executor.submit(self.layer._get_slice, request)
        task.add_done_callback(self._on_slice_done)

    def _on_slice_done(self, task: Future[ImageSliceResponse]) -> None:
        self._set_slice(task.result())

    @ensure_main_thread
    def _set_slice(self, response: ImageSliceResponse) -> None:
        ...
```

Most of the internal guts of slicing, such as the implementation of `_get_slice` and
`_make_slice_request`, are unchanged from the approach proposed here.

The main advantage of this approach is that the main consumer of the slice response (i.e. vispy)
is the one driving slicing, which allows us to better define input and output types and
avoid unnecessary type inheritance.

In the example code above, we avoid a tight dependency on `Qt` using `@ensure_main_thread`.
But another advantage of this general approach is that we could rely on tighter dependencies
between vispy and Qt, which may simplify the implementation and offer some performance benefits.

One disadvantage is that it becomes harder to control the execution of multiple layers being sliced.
For example, with the above implementation we can no longer wait for all layers to be sliced before
updating all the vispy layers.
Another disadvantage is that this implies that slice state should not live in the model in the future,
which might cause issues with things like selection that may depend on that state.


## Discussion

- [Initial announcement on Zulip](https://napari.zulipchat.com/#narrow/stream/296574-working-group-architecture/topic/Async.20slicing.20project).
    - Consider (re)sampling instead of slicing as the name for the operation discussed here.  
- [Problems with `NAPARI_ASYNC=1`](https://forum.image.sc/t/even-with-napari-async-1-data-loading-is-blocking-the-ui-thread/68097/4)
    - The existing experimental async code doesn't handle some seemingly simple usage.
- [Remove slice state from layer](https://github.com/napari/napari/issues/4682)
    - Feature request to remove slice/render state from the layer types.
    - Motivated by creating multiple slices of the same layers for multiple canvases.
    - Decision: postpone.
        - There are too many behaviors that depend on slice state (e.g. `Layer._get_value`) and it's easy enough to set it after an async slicing task is done, so leave it for now.
        - This work should simplify removing it in the future.
- [Draft PR on andy-sweet's fork](https://github.com/andy-sweet/napari/pull/16)
    - Initial feedback before sharing more widely.
- Define a class that encapsulates the slicing bounding box in world coordinates.
    - [Define a class that only encapsulates the dims and camera info needed for slicing](https://github.com/napari/napari/pull/4892/files#r935771292)
    - [Replace `point` and similar with `bbox`](https://github.com/napari/napari/pull/4892/files#r935877117)
    - The idea here is to collapse some of the existing input slicing state like `point` and `dims_displayed` into one class that describes the bounding box for slicing in world coordinates.
    - Decision: postpone.
        - This can be done independently of this work and the order is not important enough to slow this work down. 
- [Replace layer slice request and response types with a single slice type](https://github.com/napari/napari/pull/4892/files#r935773848)
    - Motivation is make it easier to add a new layer type by reducing the number of class types needed. 
    - Some disagreement here as one type for two purposes (input vs. output) seems confusing.
    - Keeping the request and response types separate may also allow us to better target alternatives to vispy.
    - Decision: keep request and response types separate for now.
        - Adding a new layer type is not a frequent occurrence.
        - Having one class may reduce the number of classes, but implementing a new layer type is already quite complex.
- [Slicing in a shader](https://github.com/napari/napari/pull/4892/files#r935322222)
    - How to handle future cases where layer data has been pre-loaded onto the GPU and we want to slice in the shader?
    - Less about large/slow data, and more about not blocking approaches to handling small/fast data efficiently.
    - May be particularly useful for multi-canvas.
    - Decision: postpone.
        - Keep related types private or vague to not blocking this in the future.
- [Support existing usage and plugins that depends on synchronous slicing](https://github.com/napari/napari/pull/4892/files#r934961214)
    - Main example is [the napari-animation plugin](https://www.napari-hub.org/plugins/napari-animation), but there may be others.
    - This was explored [^pull-4969] with the following main findings.
        - Forcing synchronous slicing when combining the prototype and `napari-animation` seems to work for the basic examples.
        - We only need to ensure synchronous slicing before using `Viewer.screenshot`, as keyframes just capture the value of `current_step` and similar.
        - `screenshot` is only called a few times in `napari-animation`, so wouldn't require large changes if asynchronous slicing was the default.
    - Decision: always have some way to force synchronous slicing.
        - But no need to support synchronous slicing by default, which is harder to implement.
- Should `Dims.current_step` (and `corner_pixels`) represent the last slice position request or the last slice response?
    - With sync slicing, there is no distinction.
    - With async slicing, `current` is ambiguous.
    - [Initial small consensus around last request](https://github.com/napari/napari/pull/4892/files#r935336654)
    - Decision: last request.
        - The implementation is simpler if this represents the last request.
        - Can introduce an additional public attribute later if last response is needed.

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

[^issue-2862]: napari issue 2862, <https://github.com/napari/napari/issues/2862>

[^pull-4334]: napari pull request 4334, <https://github.com/napari/napari/pull/4334>

[^pull-4969]: napari pull request 4969, <https://github.com/napari/napari/issues/4969>

[^vispy-faq-threads]: Vispy FAQs: Is VisPy multi-threaded or thread-safe?, <https://vispy.org/faq.html#is-vispy-multi-threaded-or-thread-safe>

[^slice-class-diagram]: napari slicing class and state dependency diagram, <https://raw.githubusercontent.com/andy-sweet/napari-diagrams/main/napari-slicing-classes.drawio.svg>

[^prototype-pr]: A PR that implements a prototype for the approach described here, <https://github.com/andy-sweet/napari/pull/21>

[^tracking-issue]: The tracking issue for this work, <https://github.com/napari/napari/issues/4795>
## Copyright

This document is dedicated to the public domain with the Creative Commons CC0
license [^cc0]. Attribution to this source is encouraged where appropriate, as per
CC0+BY [^cc0-by].
