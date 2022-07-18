(nap-4-async-slicing)=

# NAP-4: asynchronous slicing

```{eval-rst}
:Author: Andy Sweet <andrewdsweet@gmail.com>, Jun Xi Ni, Eric Perlman, Kim Pevey
:Created: 2022-06-23
:Status: Draft
:Type: Standards Track
```

## Abstract

Slicing a layer in napari generates a partial view of the layer.
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
When a dimension slider is moved, napari waits to slice each layer
before updating the canvas. When slicing layers is slow, this blocking behavior
makes navigating data difficult and napari may be reported as not responding
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
	- Once slider is moved, wait before performing slicing operation, and cancel any prior pending operations (i.e. be lazy).
	- If we can reliably infer that slicing will be fast (e.g. data is a numpy array), consider skipping this delay.
- P0. When slicing fails, I am notified so that I can understand what went wrong.
    - May want to limit the number of notifications (e.g. lost network connection for remote data).
- P1. When moving a dimension slider and the slice doesn’t immediately load, I am notified that it is being generated, so that I am aware that my action is being handled.
	- Need a visual cue that a slice is loading.
	- Show visual cue to identify the specific layer(s) that are loading in the case where one layer loads faster than another.


#### 2. Clean up slice state and logic in layers

- P0. Encapsulate the slice request and response state for each layer type, so that I can quickly and clearly identify those.
	- Minimize number of (nested) classes per layer-type (e.g. `ImageSlice`, `ImageSliceData`, `ImageView`, `ImageLoader`).
	- Capture the request state from the `Dims` object.
	- Capture the response state that vispy needs to display the layer in its scene (e.g. array-like data, scene transform, style values).
- P0. Simplify the program flow of slicing, so that developing and debugging against allows for faster implementation. 
	- Reduce the complexity of the call stack associated with slicing a layer.
	- The implementation details for some layer/data types might be complex (e.g. multi-scale image), but the high level logic should be simple.
- P1. Move the slice state off the layer, so that its attributes only represent the whole data.
	- Layer may still have a function to get a slice.
	- May need alternatives to access currently private state (e.g. 3D interactivity), though doesn't necessarily need to be in the Layer. E.g. a plugin with an ND layer, that gets interaction data from 3D visualization , needs some way to get that data back to ND.
- P2. Store multiple slices associated with each layer, so that I can cache previously generated slices.
	- Pick a default cache size that should not strain most machines (e.g. 0-1GB).
	- Make cache size a user defined preference.


#### 3. Measure slicing latencies on representative examples

- P0. Define representative examples that currently cause *desirable* behavior in napari, so that I can check that async slicing does not degrade those.
	- e.g. 3D image layer that fits in RAM, but not in VRAM.
- P0. Define representative examples that currently cause *undesirable* behavior in napari, so that I can check that async slicing improves those.
	- e.g. 3D points layer that fits in RAM.
    - e.g. 3D image layer that does not fit on local storage.
- P0. Define slicing benchmarks, so that I can understand if my changes impact overall timing or memory usage.
	- E.g. Do not increase the latency of generating a single slice more than 10%.
	- E.g. Decrease the latency of dealing with 25 slice requests over 1 second by 50%.
- P1. Log specific slicing latencies, so that I can summarize important measurements beyond granular profile timings.
    - E.g. Decrease the time spent processing the slider position move by 50%.
	- Latency logs are local only (i.e. not sent/stored remotely).
	- Add an easy way for users to enable writing these latency measurements.


### Non-goals

To help clarify the scope, we also define some things that were are not explicit goals of this project and give some insight into why they were rejected.

- Make a single slicing operation faster.
	- Can be done independently of this work.
- Improve slicing functionality.
	- For example, handling out-of-plane rotations in 3D+ images.
	- Can be done independently of this work.
- Toggle the async setting on or off, so that I have control over the way my data loads.
    - May complicate the program flow of slicing.
- When moving a dimension slider and the slice doesn’t immediately load, show of a low level of detail version of it.
	- Requires a low level of detail version to exist.
	- Should be part of a to-be-defined multi-scale project.
- Store multiple slices associated with each layer, so that I can easily implement a multi-canvas mode for napari.
	- Should be part of a to-be-defined multi-canvas project.
	- Solutions for goal (2) should not block this in the future.
- Open/save layers asynchronously.
    - More related to plugin execution.
- Lazily load parts of data based on the canvas' current field of view.
    - An optimization that is dependent on specific data formats (e.g. tiled image).
- Identify and assign dimensions to layers and transforms.
	- Should be part of a to-be-defined dimensions project.
	- Solutions for goal (2) should not block this in the future.
- Thick slices of non-visualized dimensions.
	- Currently being prototyped [^pull-4334].
	- Solutions for goal (2) should not block this in the future.
- Keep the experimental async fork working.
	- Nice to have, but should not put too much effort into this.

    
## Related work

As this project focuses on re-designing slicing in napari,
this section contains information on how slicing in napari currently works.


### Existing slice logic

Slicing in napari currently works roughly as follows.

- A dimension slider is moved, which emits the `Dims.events.current_step` event.
- `ViewerModel._update_layers` calls `Layer._slice_dims` for each layer.
- `Layer._slice_dims` updates some layer slice state (e.g. `_dims_point`, `_ndisplay`).
- `Layer._slice_dims` calls `Layer._update_dims`, which may update some other state (e.g. `_ndim`, `_transforms`).
- `Layer._update_dims` calls `Layer.refresh`, which calls `Layer._set_view_slice`.
- `Layer._update_dims` emits `Layer.events.set_data`, which calls `VispyBaseLayer._on_data_change`
- `VispyBaseLayer._on_data_change` updates the vispy node with the updated sliced state.

We can also depict a simplified version of this as a series of tasks that all run on the main thread.

![](https://i.imgur.com/Bb1TyTM.png)

Note that redrawing the slider position and canvas occur after all the layers have been sliced, likely because those are being queued on the main event loop that also runs on the main thread.

Each subclass of `Layer` has its own type-specific implementation of `_set_view_slice`, which uses the updated dims/slice state in combination with `Layer.data` to generate and store sliced data.

Similarly, each subclass of `VispyBaseLayer` has its own type-specific implementation of `_on_data_change`, which uses the new sliced data in the layer, may post-process it and then passes it to vispy to be rendered on the GPU.

The connection between `Layer.events.set_data` and `VispyBaseLayer._on_data_change` is what causes `Layer.refresh` to cause vispy to update the canvas with the latest sliced data for a layer.

In addition, some slice state is mutated by `Layer._update_draw`, which is called by `QtViewer.on_draw` which is called whenever the vispy canvas updates due to new data or a change in field of view (e.g. caused by pan or zoom). This state is mostly used when slicing multi-scale images, but may be generally useful for sending partial or low-level of detail views of the data.


### Existing slice state

It's important to understand what state is currently used for slicing in napari. Ideally, we want to encapsulate this state into an immutable slice request and response, rather than keep it on the layer as mutable state, some of which may be read by the vispy layer when slicing is done. This is especially important for asycnchronous slicing because the main thread may mutate this state while slicing is occurring, resulting in unpredictable and potentially unsafe behavior.

- `Layer`
    - `data`: array-like, the full data that will be sliced
    - `corner_pixels`: `Array[int, (2, ndim)]`, used for multi-scale images only
    - `scale_factor`: `int`, converts from canvas to world coordinates based on canvas zoom
    - `loaded`: `bool`, only used for experimental async image slicing
    - `_transforms`: `TransformChain`, transforms data coordinates to world coordinates
    - `_ndim`: `int`, the data dimensionality
    - `_ndisplay`: `int`, the display dimensionality (either 2 or 3)
    - `_dims_point`: `List[Union[numeric, slice]]`, the current slice position in world coordinates
    - `_dims_order`: `Tuple[int]`, the ordering of dimensions, where the last dimensions are visualized
    - `_data_level`: `int`, the multi-scale level currently being visualized
    - `_thumbnail`: `ndarray`, a small 2D image of the current slice
    
- `_ImageBase`
    - `_slice`: `ImageSlice`, contains a loader, and the sliced image and thumbnail
        - lots of complexity encapsulated here and other related classes like `ImageSliceData`
    - `_empty`: `bool`, True if slice is an empty image, False otherwise (i.e. hasn't been filled by exp async slicing yet?)
    - `_should_calc_clims`: `bool`, if True reset contrast limits on new slice
    - `_keep_auto_contrast`: `bool`, if True reset contrast limits on new data/slice
        
- `Points`
    - `__indices_view` : `Array[int, (-1,)]`, indices of points (i.e. rows of `data`) that are in the current slice/view
        - lots of private properties derived from this like `_indices_view` and `_view_data`
    - `_view_size_scale` : `Union[float, Array[float, (-1)]]`, used with thick slices of points `_view_size` to make out of slice points appear smaller
    - `_round_index`: `bool`, used to round data slice indices for all layer types except points
    - `_max_points_thumbnail`: `int`, if more points than this in slice, randomly sample them
    - `_selected_view`: `list[int]`, intersection of `_selected_data` and `_indices_view`, could be a cached property

- `Shapes`
    - `_data_view`: `ShapeList`, container around shape data

    - `ShapeList`
		- `_slice_key`: `list(int)`, current slice key
		- `_mesh`: `Mesh`, container to store concatinated meshes from all shapes
		- `shapes`: `list(Shape)`, list of shapes
		- `_displayed`: `Array[bool, (len(shapes))]`, mask to identify which shapes intersect current slice_key.
		- `displayed_vertices`, `Array[float, (N,2)]`, subset of vertices to be shown
		- `displayed_index`, `Array[int, (N)]`, index values corresponding to (z-order object layering) `displayed_vertices`

	- `Shape` (and subclasses... `PolygonBase`, `Polygon`, etc.)
		- `slice_key`: `list[int]`, min/max of non-displayed dimensions

	- `Mesh`
	    - Data to be shown
			- `displayed_triangles`: `Array[int, (N,3)]`, triangles to be drawn
			- `displayed_triangles_index`: `Array[int, (N)]`
			- `displayed_triangles_colors`: `Array[float, (N,4)]`, per triangle color
		- Shape meshes generated at shape insertion
		    - `vertices`, `vertices_centers`, `vertices_offsets`, `vertices_index`,
		      `triangles`, `triangles_index`, `triangles_colors`, `triangles_z_order`

- `Vectors`
    - `_view_data`: `(M, 2, 2) array`:
        The start point and projections of N vectors in 2D for vectors whose
        start point is in the currently viewed slice. Subset of `data`
    - `_view_indices`: `(1, M) array`:
        indices for the M in view vectors (indices for subsetting `data`)
    - `_view_alphas`: `(M,) or float`:
        relative opacity for the M in view vectors
    - `_view_faces`:  `(2M, 3) or (4M, 3) np.ndarray`:
        indices of the `_mesh_vertices` that form the faces of the M in view vectors.
        Shape is (2M, 2) for 2D and (4M, 2) for 3D. 
		* Subset of `_mesh_triangles`
    - `_view_vertices`: `(4M, 2) or (8M, 2) np.ndarray`:
        the corner points for the M in view faces. Shape is (4M, 2) for 2D and (8M, 2) for 3D.
		* Subset of `_mesh_vertices`
    - `out_of_slice_display`: `bool`:
        If True, renders vectors not just in central plane but also slightly out of slice
        according to specified point marker size.

	- Note: `_view_faces` and `_view_vertices` require:
		- `_mesh_vertices` - output from `generate_vector_meshes`, not specific to slice
		- `_mesh_triangles` - output from `generate_vector_meshes`, not specific to slice


## Detailed description

This project aims to perform slicing on layers asynchronously.
To do so, we introduce a few new types to encapsulate state that is critical
to slicing, some new methods that redefine the core logic of slicing,
and show how these new things integrate into napari's existing design.

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

#### Response

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
        requests = {layer: layer._make_slice_request(dims) for layer in layers}
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
we anticipate that this might grow, further motivating a distinct type.

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

This section describes the ways in which the NAP affects backward
compatibility, including both breakages and decisions that better support
backward compatibility.


## Future work

This section describes work that is out of scope for the NAP, but that the
NAP might suggest, or that the NAP author envisions as potential future
expansion of the work or related work.


## Alternatives

If there were any alternative solutions to solving the same problem, they
should be discussed here, along with a justification for the chosen
approach.


## Discussion

This section may just be a bullet list including links to any discussions
regarding the NAP, but could also contain additional comments about that
discussion:

- This includes links to discussion forum threads or relevant GitHub discussions.


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


## Related technical details

### Methods and properties on vector layer used by slicing

* **METHOD**:`Vectors._set_view_slice()`: Sets the view given the indices to slice with.
    * Uses
        * `_slice_indices`: property (see below)
        * `_displayed_stored` - I think this should just be removed altogether. I don't see a purpose. 
        * `_dims_displayed`
        * `_mesh_vertices`
        * `_mesh_triangles`

    * Calls
        * `slice_data()`: to generate the `indices` and `alphas` (see below)
            * Which uses the property `_slice_indices`
        * `generate_vector_meshes()`: If the mesh hasn't already been generated, it will create it to get the vertices and triangles (see below)
    * Sets: 
        * `_view_data` 
        * `_view_indices`
        * `_view_alphas`
        * `_view_faces`
        * `_view_vertices`

* **METHOD**: `Vectors._slice_data()`: Determines the slice of vectors given the indices.
    * Used by: `_set_view_slice`
    * Uses `_slice_indices` 

* **METHOD**: `_vector_utils.generate_vector_meshes()`: creates the vertices and faces on which to display the vectors (for all the data, not just the current slice)
    * Uses:
        * `_data`
            * is subset using:
                * `_dims_displayed`
                    * `_ndisplay` (connected to event)  Number of visualized dimensions
                    * `_dims_order` List of dims as indices (if ndim=3, dims_order=[0, 1, 2])
                    * if there are fewer dims displayed (ndisplay) than the size of the data (dims_order), then napari will automatically grab the last 2 dims to visualize (dims_displayed)
                        * `_ndim` Number of dims of the data itself
        * `edge_width`
        * `length`
    * Output:
        * `vertices`
        * `triangles`

* **PROPERTY**: `Vectors.out_of_slice_display`: bool: 
    * renders vectors slightly out of slice, accounts for vectors which are "slightly-out-of-frame"
    * has a setter which calls
        * `self.events.out_of_slice_display()`
        * `self.refresh()`

* **PROPERTY**: `Base._slice_indices`: (D, ) array: 
    * slice indices into data coordinates
    * complex getter
        * Uses: 
            * `_dims_not_displayed`
            * `ndim`
            * `_ndisplay`
            * `_dims_point`
        * Could use (via `if` statement):
            * `_data_to_world.inverse`
            * `utils.transforms.Affine`
            * `_dims_displayed`
