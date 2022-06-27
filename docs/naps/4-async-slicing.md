(nap-4-async-slicing)=

# NAP-4: asynchronous slicing

```{eval-rst}
:Author: Andy Sweet <andrewdsweet@gmail.com>, Jun Xi Ni, Eric Perlman
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
	- e.g. Medium 3D image layer.
	- Small: all data fits in VRAM (i.e. < 1GB).
	- Medium: all data fits in RAM but not in VRAM (e.g. 1GB < x < 8GB).
- P0. Define representative examples that currently cause *undesirable* behavior in napari, so that I can check that async slicing improves those.
	- e.g. Large local 3D points layer.
	- e.g. Huge remote 3D image layer.
	- Large: all data fits on local storage but not in RAM (e.g. 8GB < x < 128GB).
	- Huge: all data does not fit on local storage (e.g. > 128GB).
- P0. Define slicing benchmarks, so that I can understand if my changes impact overall timing or memory usage.
	- E.g. Do not increase the latency of generating a single slice more than 10%.
	- E.g. Decrease the latency of dealing with 25 slice requests over 1 second by 50%.
- P1. Log specific slicing latencies, so that I can summarize important measurements beyond granular profile timings.
	- Latency logs are local only (i.e. not sent/stored remotely).
	- Add an easy way for users to enable writing these latency measurements.


### Non-goals

To help clarify the scope, we also define some things that were are not explicit goals of this project and give some insight into why they were rejected.

- Make a single slicing operation faster.
	- The slicing code should mostly remain unchanged.
	- Useful future work, that may be made easier by changes here.
	- Scope creep: can be done independently on this work.
- Improve slicing functionality.
	- For example, handling out-of-plane rotations in 3D+ images.
	- The slicing code should mostly remain unchanged.
	- Useful future work, that may be made easier by changes here.
	- Scope creep: can be done independently on this work.
- Toggle the async setting on or off, so that I have control over the way my data loads.
    - May complicate the program flow of slicing.
- When moving a dimension slider and the slice doesn’t immediately load, show of a low level of detail version of it, so that I can preview what is upcoming.
	- Requires a low level of detail version to exist.
	- Scope creep: should be part of a to-be-defined multi-scale project.
- Store multiple slices associated with each layer, so that I can easily implement a multi-canvas mode for napari.
	- Scope creep: should be part of a to-be-defined multi-canvas project.
	- Solutions for goal (2) should not block this in the future.
- Open/save layers asynchronously.
    - More related to plugin execution.
- Lazily load parts of data based on the canvas' current field of view.
    - An optimization that is dependent on specific data formats (e.g. tiled image).
- Identify and assign dimensions to layers and transforms.
	- Scope creep: should be part of a to-be-defined dimensions project.
	- Solutions for goal (2) should not block this in the future.
- Thick slices of non-visualized dimensions.
	- Scope creep: currently being prototyped [^pull-4334].
	- Solutions for goal (2) should not block this in the future.
- Keep the experimental async fork working.
	- Nice to have, but should not put too much effort into this.
	- Do not delete some existing code, which may be moved into vispy (e.g. VispyTiledImageLayer).

    
## Related work

As this project focuses on re-designing slicing in napari, this section contains information on how slicing in napari currently works.


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

## Detailed description

This section should provide a detailed description of the proposed change. It
should include examples of how the new functionality would be used, intended
use-cases, and pseudocode illustrating its use.


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
[^cc0-by]: <https://dancohen.org/2013/11/26/cc0-by/>
[^issue-792]: napari issue 792, <https://github.com/napari/napari/issues/792>
[^issue-1353]: napari issue 1353, <https://github.com/napari/napari/issues/1353>
[^issue-1574]: napari issue 1574, <https://github.com/napari/napari/issues/1574>
[^issue-1775]: napari issue 1775, <https://github.com/napari/napari/issues/1775>
[^issue-2156]: napari issue 2156, <https://github.com/napari/napari/issues/2156>
[^pull-4334]: napari pull request 4334, <https://github.com/napari/napari/pull/4334>

## Copyright

This document is dedicated to the public domain with the Creative Commons CC0
license [^id3]. Attribution to this source is encouraged where appropriate, as per
CC0+BY [^id4].
