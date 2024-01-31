(nap-9)=

 # NAP-9 — Multiple Canvases

```{eval-rst}
:Authors: Ashley Anderson <aandersoniii@chanzuckerberg.com>, Wouter-Michiel Vierdag, Lorenzo Gaifas
:Created: 2023-08-04
:Status: Draft
:Type: Standards Track
```

## Definitions

In order to facilitate discussion - this NAP will use the following definitions.

*Viewer* - Currently maps basically 1:1 to the napari application main window, including canvas(es), dims sliders, layer list, layer controls, and dock widgets. Related is the `ViewerModel`, a class in napari that maintains the state of the Viewer.

*Canvas* - The main napari data view (2D rectangle) where 2D or 3D slice data is displayed, and additional data views displayed within a single Viewer window. This NAP is in part an attempt to formaize this concept, and as noted proposes a related `CanvasModel` class in napari to hold the state of a Canvas.

*Layer* - The base unit of the napari image data model. A `ViewerModel` maintains an ordered list of Layers that it may display on its Canvas.

*Layer Slice* - A subset of data from a Layer, reduced to 2D or 3D (via slicing) for visualization. This again is a general conecpt that this NAP proposes to formalize as a napari class (or set of classes paired with the Layer types).

*Visual* - The corresponding visual representation of a Layer Slice displayed on a Canvas. The specific visual is determined by the Layer (and Layer Slice) type.

Note that [VisPy](https://vispy.org/) (the current backend for all napari visuals) has its own specific definitions for some of these or related concepts, such as [`Canvas`](https://vispy.org/api/vispy.app.canvas.html#module-vispy.app.canvas) and [`ViewBox`](https://vispy.org/api/vispy.scene.widgets.viewbox.html#module-vispy.scene.widgets.viewbox). Where necessary to refer to these concepts in this NAP (or discussion), such concepts will be qualified accordingly (for example: "a VisPy Canvas").

## Abstract

Current napari architecture supports a single canvas (viewbox) per viewer (window). Simultaneously showing multiple views of the same data generally necessitates opening an entirely new napari viewer window or low-level work with Qt widgets and private napari APIs. This wastes resources (primarily memory) and complicates interaction.

## Motivation and Scope
The ability to view n-D data from multiple perspectives (or different data from the *same* perspective -- for example side-by-side segmentations) is a common feature request, and has proved useful in many other tools for data exploration and analysis. Here is a sampling of issues requesting support and discussing potential implementations:

* [#5348](https://github.com/napari/napari/issues/5348) Multicanvas viewer
* [#2338](https:////github.com/napari/napari/issues/2338) Multicanvas API Thoughts
* [#760](https:////github.com/napari/napari/issues/760) Linked multicanvas support
* [#662](https:////github.com/napari/napari/issues/662) Linked 2D views
* [#561](https:////github.com/napari/napari/issues/561) multicanvas grid display for layers in Napari
* [#1478](https:////github.com/napari/napari/issues/1478) Orthogonal viewer plugin

Several plugins and examples have been created to address these limitations, for example:
* [napari-3d-ortho-viewer](https://github.com/gatoniel/napari-3d-ortho-viewer/tree/main)
* [multiple viewer widgets example](https://napari.org/stable/gallery/multiple_viewer_widget.html#sphx-glr-gallery-multiple-viewer-widget-py)

This document is intended to cover what [#5348](https://github.com/napari/napari/issues/5348) refers to as "True Multicanvas".

Providing native support in napari would allow developers to more easily create these experiences, enable interoperability between such plugins, and improve performance.

### Out of Scope
* Improvements to VisPy to support multiple views of the same `SceneGraph` (sharing data, saving VRAM) - for relevant discussion start with [vispy/#1992](https://github.com/vispy/vispy/issues/1992).
* For now, canvas arrangement (for example: tiling behavior) will be handled in the view only (left to Qt or custom Qt widgets). Making this state (de)serializable is out of scope for this project, but may be relevant when implementing a “savable viewer state” feature.
* Normalizing slice data for different layer types, though this may benefit in the course of this work.
* Specific UI implementations will be explored as part of this work, but I expect UX and UI will be formalized later (possibly in a separate NAP).
* Supporting alternative frontend (Qt) and backend (Vispy) frameworks. While this work should not make such tasks more difficult in the future, explicit consideration is out-of-scope until further progress is made in these areas.
* [Non-goals also in NAP-3](https://napari.org/dev/naps/3-spaces.html#non-goals) are related but also considered out-of-scope here
    * Separation of rendering information from the Layer(Data) model
    * Window state restoration

## Detailed Description

### Requirements
* The application data model (`ViewerModel` + Layers) shall support multiple canvases.
* The application shall natively display multiple canvases simultaneously.
    * There shall be a minimum of one canvas (current status) per viewer.
* All canvases shall share a common layer list and (unsliced) layer data.
* Each canvas shall have independent:
    * Data slicing
    * Camera (zoom, center)
    * Dimensionality (2D/3D display)
    * Layer visibility
* The implementation should minimize changes to the existing public API.
* The napari application (`ViewerModel`) shall maintain a concept of a single “active” (currently focused) canvas.
    * Alternatively, there could be a “main” canvas that does not change (“main” and “active” could even be simultaneously supported).
    * There will be no possibility of a viewer with no canvases.
* Users shall be able to add, remove, and (maybe[^maybe-rearrange]) rearrange canvases.

[^maybe-rearrange]: Exact UI/UX may is yet to be decided, see [UI Architecture](#UI-Design-and-Architecture) for some discussion.

### Design Considerations & Decisions
Part of this design document is intended to capture the desired behavior and prevent scope creep. At the extreme “multiple canvases” can be achieved with “multiple viewers”. Therefore we need to draw a line somewhere to differentiate a “canvas” from a “viewer”. [^napari-lite]

> TODO: add rough CanvasModel definition here [name=Ashley A]

[^napari-lite]: A lightweight "canvas" might be relevant to the implementation of ["napari-lite"](https://github.com/napari/napari/issues/5940).

An important consideration is to minimize breaking changes to the public napari API. While napari is still pre-1.0, there is already a healthy developing ecosystem of plugins, scripts, and users. Changes to the API may be necessary and should be made if they constitute improvements, but should be minimized and well documented.

The concept of an "active" canvas will work in service of minimizing API changes. This will allow existing APIs on the main Viewer/ViewerModel to remain and simply delegate to the active canvas.

In addition to maintaining the model-view-controller (MVC) architecture of napari, this proposal aims to maintain or improve decoupling of the UI framework (currently Qt), the visualization library (currently VisPy), and the napari core code.

> TODO: add a list open questions and key decisions here [name=Ashley A]
> * selection state

### Architecture
napari architecture is based on the MVC pattern. The model layer comprises a `ViewerModel` and a list of Layer models (subclasses of a base `Layer`). There are seven layer types, each with a corresponding view type. Currently models and views are paired 1:1, and the correlation is stored in a map (`layer_to_visual`) on the `VispyCanvas`. Figure 1 shows the class relationships for the base model types and the Image layer types (for brevity - other layer types have similar connectivity).

```{image} _static/multicanvas-napari-architecture-today.png
---
name: fig-1
---
Fig. 1: napari architecture today.
```

Figure 2 shows proposed changes (in orange) to the architecture to support multiple canvases. The new architecture is still following the MVC pattern. Again, this diagram only includes the Image layer type. Here is a summary of the planned changes:
* Slice state will be moved off the layer as necessary, into new `LayerSlice` classes for each layer type
    * This will be different for each Layer type - unifying the structure in the process may be a secondary benefit but is not the goal
* Each VispyCanvas will hold a reference to a dedicated model class (`CanvasModel`)
    * dims will move from `ViewerModel` -> `CanvasModel`
    * camera will move from `ViewerModel` -> `CanvasModel`
    * `layer_to_slice` will map each Layer (global list) to a `LayerSlice` (one per `CanvasModel`)
* `ViewerModel` will own a *list* of `CanvasModel` objects
* `QtViewer` will own a *list* of `VispyCanvas` objects
* `VispyLayer` subclasses will hold references to their `Layer` (for rendering information) as well as a `LayerSlice` (for data)
* The LayerSlicer will need to update the sync and async callbacks[^async-only] to pass a canvas parameter where the resulting sliced data will be stored, rather than storing it on the layer itself. Slice task cancellation logic will need to be revisited accordingly.
    * async callback: `LayerSlicer._on_slice_done`
    * sync callback: `Layer._slice_dims`
* Callbacks (interaction, events) will need to be specifically connected to individual `CanvasModel` objects where relevant (dims, camera) rather than the `ViewerModel`.


```{image} _static/multicanvas-napari-architecture-tomorrow.png
---
name: fig-2
---
Fig. 2: napari architecture tomorrow, with proposed changes from this NAP highlighted in orange.
```

> Note: in both diagrams, the `on_draw()` method on the `VispyCanvas` breaks MVC convention (view layer talks directly to the model layer). This is a separate/known issue and I believe is mostly only true for multiscale image layers right now. Changing this is considered out of scope for this project at this time.

Slice state for each layer is currently stored on the Layer model. Again, see [NAP-4 for previous discussion](https://napari.org/stable/naps/4-async-slicing.html#existing-slice-state). This NAP proposes to move this state off the Layer instance, into a specific Layer Slice instance. This is what will allow multiple slices of a single layer to be visualized simultaneously. *Table 1* lists the attributes related to slice state that will be moved in this work from each Layer class into corresponding Layer Slice classes.

> Note: some layers include "seleciton" information (Points, Shapes, and Tracks). These will be considered as Layer-level concepts, unrelated to the proposed changes.

***Table 1** - Layer attributes that hold slice data. These attributes will be moved from the Layer onto individual Layer Slice objects (one Layer Slice per Layer per Canvas).*
> TODO: make this a list or figure out a way to center it [name=Ashley A]

| Layer Class | Slice Attributes |
| -------- | -------- |
| Base | `_slice_input`
| | `_slice_indices`
| Image, Labels | `_data_view`
|| `_empty`
| Points | `_view_size_scale`
|| `_indices_view`
| Surface | `_view_vertex_values`
|| `_view_vertex_colors`
|| `_data_view`
|| `_view_faces`
| Vectors | `_view_data`
|| `_view_face_color`
|| `_view_indices`
|| `_view_alphas`
| Tracks | `_view_data`
|| `_view_size`
|| `_view_symbol`
|| `_view_edge_width`
|| `_indices_view`
|| `_drag_start`
| Shapes | `_data_dict`
|| `_data_view`

> TODO: add rough class definition(s) for Layer Slice(s)
> Also - the various `_<Layer>SliceResponse` classes introduced by async slicing may already fill much of this role. Another option to consider here is to codify a protocol ([`typing.Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)) for these classes. Even this protocol may not be necessary - early prototypes use these classes as-is with minimal modifications.
> [name=Ashley A]

### UI Design and Architecture
Specific UI design and architecture remains to be determined. This will be explored as part of step 4 in the [Implementation Plan](#Implementation). UI design needs additional refinement and exploration, and this is expected to continue after basic/core implementation propsed in this NAP is complete. UI changes may also be described in a separate NAP along with a discussion of convenience functions and affordances for common operations. Some placeholder or experimental code will be used in the meantime as a prototype implementation.

Some open questions here are (for example):
* Should each canvas also have visible dims sliders, or can we keep one set of dims sliders that changes based on the active (selected) canvas?
* What kind of cross-reference displays or tools should there be?
    * through-plane slice indicators
    * three-point slice definition
* What kinds of camera-linking should be supported?
    * orthogonal
    * stereoscopic

Beyond showing a grid of canvases, it would be nice for individual canvases to be:
* Resizable
* Reorderable
* Re-tileable (for example, changing number of rows and columns to tile)
* Maybe: Maximized, stacked, and minimized (e.g. with tabs)

Here are some Qt classes that may provide a sound base for multicanvas UI implementation:
* `QDockWidget`, with the main window being modified to allow dock widget nesting (`dockNestingEnabled`). This may require the fewest modifications to the existing Qt viewer. Allowing widgets to be undocked would make this extremely flexible, but possibly also confusing.
* `QMdiArea` (“multiple document interface”) satisfies most of these requirements, and should be customizable to satisfy them all. This would offer extreme flexibility of layout.
* `GridLayout` would likely provide a quite simple but otherwise inflexible solution. For example this may make independent resizing of canvases difficult.

## Related Work
See other image viewers for examples for multiple canvases (mostly demonstrating orthogonal views):
* [3D Slicer](https://www.slicer.org/)
* [Orthogonal views in ImageJ and Imaris](https://www.youtube.com/watch?v=94d8sHMP_w8)
* [OHIF/Cornerstone.js](https://www.cornerstonejs.org/live-examples/crosshairs)
* [neuroglancer](https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B8e-9%2C%22m%22%5D%2C%22y%22:%5B8e-9%2C%22m%22%5D%2C%22z%22:%5B8e-9%2C%22m%22%5D%7D%2C%22position%22:%5B2914.500732421875%2C3088.243408203125%2C4045%5D%2C%22crossSectionScale%22:3.762185354999915%2C%22projectionOrientation%22:%5B0.31435418128967285%2C0.8142172694206238%2C0.4843378961086273%2C-0.06040274351835251%5D%2C%22projectionScale%22:4593.980956070107%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://neuroglancer-public-data/flyem_fib-25/image%22%2C%22tab%22:%22source%22%2C%22name%22:%22image%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://neuroglancer-public-data/flyem_fib-25/ground_truth%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%2221894%22%2C%2222060%22%2C%22158571%22%2C%2224436%22%2C%222515%22%5D%2C%22name%22:%22ground-truth%22%7D%5D%2C%22showSlices%22:false%2C%22layout%22:%224panel%22%7D)

## Implementation

1. Introduce minimally disruptive `_canvases` attribute on the ViewerModel
    * Just a list (`EventedList`) of `CanvasModel` objects
    * One canvas is “active”, relevant properties on ViewerModel (`camera`, `dims`) are delegated to the active canvas
        * "active" canvas is just index 0 (alternatively: use `SelectableEventedList`)
    * Only the active canvas is shown (minimal modifications to `QtViewer` and `VispyCanvas`)
    * A single `QtDims` view (slicing sliders) is shown, updates depending on the active canvas
    * Adjust event callbacks such that interactions (slicing, camera movement, ndisplay toggle) only apply to the active canvas
    * Add public APIs to add/remove/rotate (change active) `_canvases`

2. Add Layer Slice classes to reduce data reslicing when switching between canvases
    * Start with `Image` and `Labels` layers
        * Move/modify/replace `set_view_slice` and `_update_slice_response` to set data on Layer Slice for associated canvas
    * Modify `VispyCanvas`
        * Get camera and dims information from associated `CanvasModel` instead of `Viewer`
        * Obtain relevant data from Layer Slice instead of Layer
    * Modify `ViewerModel` and `LayerSlicer` [^async-only]
        * Submit `CanvasModel` (or an ID) to `LayerSlicer` instead of dims directly
        * Emit `CanvasModel` (or an ID) and slice data from `LayerSlicer.ready` and/or `Layer.set_data` events
        * `VispyCanvas` will subscribe to relevant events, set data if corresponding to its own `CanvasModel`. Other canvases may also be interested in this event for example to update cross-reference overlays.

3. Update `QtViewer` and `VispyCanvas` to support multiple canvases
    * Still only displaying one canvas at a time in the main widget
    * Update main widget as `ViewerModel` “active” canvas changes, storing additional canvases and swapping them out as necessary

4. Update `QtViewer` to show multiple canvases simultaneously
    * This is exploratory work at the moment, see [UI Architecture](#UI-Design-and-Architecture) section below

[^async-only]: Depending on the timeline and prototype implementation, it may be acceptable/preferable for multi-canvas feature to rely on (currently experimental) async slicing ([see NAP-4](https://napari.org/stable/naps/4-async-slicing.html)).

## Backward Compatibility

Maintaining the proxy API on the viewer via the concepts of a main and/or active canvas will make this work mostly backward-compatible, though there will inevitably be some breaking changes. There will likely be significant breaking changes to private APIs. For example if plugins are attempting to access slice data directly from a layer instance, it may no longer be as expected. If this is a large burden, it too may be mitigated by delegating from the layer to the slice corresponding to the main or active canvas.

## Future Work

The goal of this NAP is to cover the main architectural changes to enable multi-canvas work. Future work is expected in 1) user experience, design, and GUI implementation details; and 2) consistent, ergonomic, and documented public APIs for advanced interaction with multiple canvases.

## Alternatives

### Users can open multiple napari viewers
Using multiple napari viewers does not satisfy the core user needs for multiple canvases when processing or manipulating data. Multiple viewers also wastes system resources as viewers do not communicate or share memory.

### Leave multicanvas to plugins and custom widgets

Unifying an implementation and (eventually) providing a stable multi-canvas API will save work for plugin authors, and allow more plugins to interoperate.

### Implement slices using shallow Layer copies
This is a good and reasonable alternative to the proposed implementation, and is how the [multiple viewer widgets example](https://napari.org/stable/gallery/multiple_viewer_widget.html#sphx-glr-gallery-multiple-viewer-widget-py) is implemented. This implementation also makes it easier to configure rendering/appearance per-canvas (layer visibility, colormap, etc.). However this implementation relies more on careful bookkeeping than data modeling. If this is desired functionality, layer data should be fully separated from the data view (slice and view state). Ultimately this implementation is similar to that proposed in this NAP, and could be considered along a continuum of separating layer data, slice data, and rendering configuration.

## Discussion

* [#5348](https://github.com/napari/napari/issues/5348) Multicanvas viewer
    * This is the most recent and thorough discussion multi-canvas prior to this NAP

## Copyright

This document is dedicated to the public domain with the Creative Commons CC0
license [^id3]. Attribution to this source is encouraged where appropriate, as per
CC0+BY [^id4].


[^id3]: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication,
    <https://creativecommons.org/publicdomain/zero/1.0/>

[^id4]: <https://dancohen.org/2013/11/26/cc0-by/>
