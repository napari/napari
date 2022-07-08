(nap3)=

# NAP-3 — Spaces

```{eval-rst}
:Author: Lorenzo Gaifas, brisvag@gmail.com
:Created: 2022-06-08
:Status: Draft
:Type: Standards Track
``` 


# Abstract

`napari` is currently limited to holding (and rendering) data belonging to a single, universal space (which we often refer to as *world space*). However, it is often useful to have quick and easy access to different parts of a dataset that do not belong to the same coordinate system. In these cases, forcing data to live in the same world not only makes no sense, but it can make navigating the data and interacting with the viewer slower and less intuitive.

This NAP discusses the reasons why a native napari approach would be better than the currently available workarounds. It then proposes the introduction of `spaces` as a way to manage different coordinate spaces in the same viewer.


# Motivation and scope

This NAP aims to address a few problems that arise (especially with big datasets), when working with many layers that do not necessarily belong to the same *absolute* coordinate space. For example, there is no reason to relate between the *absolute* coordinates of `image 1` and `image 2` from the same microscopy data collection, but it might be useful to quickly switch between the two to -- for example -- compare the effectiveness of a processing step on different images, or to visually inspect qualitative differences.

Currently, to do so, a user is forced to either:
- load everything into the layer list (thus "pretending" that the absolute coordinates do indeed match) and then develop a plugin or widget that manages layer visibility
- or develop a plugin or widget that manages layers externally and feeds them to `Viewer.layers` as desired
- do nothing and deal with hundreds of layers manually

While this might not seem problematic with few images, it quickly degenerates when working with big datasets, especially in 3D (for an example use case, check out [this issue comment](https://github.com/napari/napari/issues/4419#issuecomment-1113090992)).

The above workarounds have the following issues:
- *usability*: adding too many layers to the viewer makes navigating the layerlist unwieldy, a problem that cannot be solved by layer-groups when there is no reasonable way to group layers
- *usability*: forcing the creation of a plugin or widget with custom logic is one more barrier for non-developers, and possibly one more meta-object (e.g: a "dataset") that the user has to juggle around and that napari is unaware of.
- *performance*: many layers perform worse than few layers. This might be addressed separately, but is currently unresolved.
- *abstraction*: it makes no sense for things to share coordinate space when they shouldn't.
- *performance and usability*: regardless of the workaround used, reader plugins can't know about it, so a user would be forced to either first load everything into the viewer and then do something with it, significantly slowing down startup and adding one more step to opening napari
- *serialization*: without a native `napari` support, these workarounds cannot be properly serialized.

Additionally, while this was not the main goal of this NAP, people have sometimes asked for "workspaces", where different workflows can be tested in parallel [^workspaces]; `spaces` would also implicitly allow this by letting users re-use a layer in multiple spaces.

## Non-goals

- *window state*: in [#4227](https://github.com/napari/napari/issues/4227), a proposal was advanced for managing window state and layout, with the ability to re-use and restore them. While this could be conceivably be dealt with here, it is probably better to keep separate the state of the window from the representation and the data.
- *rendering/data separation*: while necessary for some of the future benefits of spaces (i.e: [](nap-3:multicanvas)), the separation of rendering and data from the currently unified `Layer` object is not in the scope of this NAP. This means that (just like now), layer won't be shareable between `Viewer`s. However, they should be shareable between `Space`s, as long as the spaces are not *rendered* at the same time (i.e: in separate `Viewer`s).

# Implementation proposal

There are a few ways to tackle this issue (see [](nap-3:alternatives)), with different upsides and downsides; the current "main" proposal is aiming to solve the issue by avoiding breaking or major api changes (which are instead required for the alternatives).

## API

`ViewerModel.spaces` would be a (selectable) `EventedList` or similar evented collection, each containing a `Space` object, with the following attributes:
- `layers`: a `layerlist` (or top-level `layergroup`, in the future)
- `camera`: a snapshot of the state of the `Camera` model (i.e: `Camera.dict()`)
- `dims`: a snapshot of the state of the `Dims` model (i.e: `Dims.dict()`)

Additionally, it would be useful and intuitive for users to mirror some of the basic `Layer` API:
- `name`: a unique name
- `metadata`: to hold any extra information about the `Space` (for example, experimental conditions, or workflow description)
- `source`: to hold the source `path` and reader plugin, in case a plugin [generated the space](nap-3:plugins).

A few more `ViewerModel` attributes are worth considering for being tracked by `State`s, depending on if we think they should be considered "global" settings, or local to the `Space`. I propose the following:
- `grid`: local, if you leave a state in grid mode, you probably want it back that way
- `scale_bar`: global
- `text_overlay`: local, as text is likely to refer to the contents of the viewer
- `overlays`: tricky; this is currently only the `InteractionBox` (and therefore should probably *not* be serialized, since it's dynamically changed when transforming layers), but might become more complicated in the future, if something like [napari/napari#3763](https://github.com/napari/napari/pull/3763) gets merged).

The remaining fields should probably not be serialized:
- `cursor`
- `help`
- `status`
- `tooltip`
- `theme`
- `title`

In the end, all `spaces` would do is effectively provide a quick way to swap some of the `ViewerModel` state in and out.

---

At the level of `ViewerModel` itself, we would have an `active_space` attribute: similar to how a layer in a `layerlist` can be active, a `Space` in the `spaces` can be active (with the important distinction that only *one* space can be active); this will determine which space is loaded into the layerlist and used to populate the canvas.
    - in a future with multi-canvas (or multi-viewer), this could be on a per-canvas (or per-viewer) basis.

## GUI

The GUI could expose this as a (searchable) dropdown menu above the layerlist, and provide shortcuts to navigate easility through spaces, such as `page-up`, `page-down`.

(nap-3:plugins)=
## Plugins

Reader/writer plugins should be able to provide/consume spaces. If unspecified, they should act on the active `Space`, which would be backwards compatible.

Widget plugins would be backwards compatible, as they simply act on the active `Space`. On the other hand, new plugins would be able to access other spaces as well, allowing for easier abstraction of "batch" workflows.


(nap-3:alternatives)=
# Alternatives

## Naming

Instead of `Space`, we could use a different name:

- `Viewer`: better conveys that viewer state is also retained. I find it confusing due to the strong coupling with the `Window` (see also [](nap-3:multiple-viewers))
- `State`: better conveys that non only layerlist state is retained. A bit generic.

(nap-3:multiple-viewers)=
## Multiple viewers, `app` interface

These problems could be also solved by allowing multiple `Viewer` objects, each with its own `ViewerModel`, by separating out the `QtViewer` logic to an `Application` level [^application].

```python
app.viewers  # list of all viewers
app.viewer  # current viewer model
app.window  # the singleton window now lives on the app rather than the viewer
```

This is in many ways equivalent to `spaces`, with object serving the same purpose but being named differently: `Application` takes place of the `Viewer` object; `Viewers` are acting as `Spaces`, with the difference that they are themselves a `ViewerModel`, rather than holding a snapshot of parts of a `ViewerModel`.

While the nomenclature is not one of the important points of this NAP, I find the use of the word `Viewer` confusing in this alternative implementation. Intuitively, for me (and, I expect, for most users) the `Viewer` is the window, regardless of the abstraction of `ViewerModel` and `Viewer` that we have on the backend. Additionally, the above changes would cause a big API change in the most basic interaction with `napari` (`app.viewer` rather than `viewer`), unless we "hide" it away (i.e: `viewer.app.viewers`), which would hurt usability and discoverability.

Additionally, there are no benefits to keeping multiple `ViewerModel` objects alive, since their purpose is simply to act as an evented model to update the GUI.

Finally, this alternative would further complicate our ability to support multiple *actual* `Viewers` with their own window, separate from each other [^multiple-viewers]. In that case, the `viewer.app.viewers` seems like a better fit and wouldn't add too much overhead to basic operations.

## Spaces, `app` interface

Alternatively, the `app` interface can be used in conjunction with the `spaces` approach, distinguishing between `Space` and `Viewer`:

```python
app = viewer.app  # shared app object between viewer
app.spaces  # spaces list, but held by the app object (and thus accessible across viewers)
viewer.active_space = app.spaces[0]  # choose which space to display in a viewer
```

This improves on the [](nap-3:multiple-viewers) approach ont the clarity of separation between `Spaces` and `Viewers`, and on the main proposal by explicitly allowing sharing spaces between viewers. 

A downside is that we lose the single point of truth for the `Spaces`. If a space is loaded in two viewers, we would need to connect events so that if something about the state is changed (such as the `Dims`) it should update the state of all the other Viewers attached to the state. This is not necessary for the layerlist and the layers, since those would *be* the same objects; in fact, this is a point in favour of using `ViewerModel` themselves to encode the state, as proposed in [](nap-3:multiple-viewers), or to at least *split out* from the `ViewerModel` the fields that would be instead held by `Spaces`.

# Backward compatibility

These changes should be backwards compatible, since they would only expand the `ViewerModel` API by adding spaces.


# Future work

## Tabbed access

As part of the "multiple viewers" proposals in the past, the idea of accessing them through tabs in the GUI was often floated [^multiple-viewers-tabbed]. The same idea can be applied to `spaces`. This is not necessarily mutually exclusive with the searchable dropdown approach: a space could be "pinned", allowing easier access through the GUI. However, the primary access should not be tabs, which would the defeat one of the goals of `spaces`: de-cluttering the GUI.

(nap-3:multicanvas)=
## Multicanvas

While multicanvas is still some ways off, this NAP can provide the basis for that functionality. For example, a future multicanvas-capable viewer could associate each `Canvas` to a `Space`; this way, we already have the machinery for multiple layer lists, as well as the ability to re-use the same layer in multiple canvases, while having a different `Camera` and `Dims` setup for the different copies (note that this would rely on the current efforts in separating the slicing logic from the `Layer` object).


# References

For the original discussion, see [napari/napari#4419](https://github.com/napari/napari/issues/4419).


# Copyright

This document is dedicated to the public domain with the Creative Commons CC0
license [^cc0]. Attribution to this source is encouraged where appropriate, as per
CC0+BY [^cc0by].

[^workspaces]: https://github.com/napari/napari/issues/4419#issuecomment-1126375339

[^application]: https://github.com/napari/napari/issues/4419#issuecomment-1129443846

[^multiple-viewers]: https://github.com/napari/napari/issues/3955

[^multiple-viewers-tabbed]: https://github.com/napari/napari/issues/3956

[^cc0]: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication,
    <https://creativecommons.org/publicdomain/zero/1.0/>

[^cc0by]: <https://dancohen.org/2013/11/26/cc0-by/>
