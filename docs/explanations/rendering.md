# Overview

This document outlines our plans for making napari's rendering non-blocking. We hope to update this document as the implementation evolves with more concrete and final details.

# Blocked UI

In May 2020 we looked into three issues related to the UI becoming blocked:

| Issue | Summary                                                        |
| ----- | -------------------------------------------------------------- |
| #845  | UI blocked by Disk or Network IO rendering multi-scale images. |
| #1300 | UI blocked rendering large in-memory images (not multi-scale). |
| #1320 | UI blocked rendering small images due to lazy computations.    |

When the UI is "blocked" napari feels slow and lags. It's not just an aesthetic
issue, manipulation of interactive UI elements like sliders becomes nearly
impossible if the framerate is low enough. If the GUI thread is blocked for long
enough you can get the "spinning wheel of death" on Macs indicating the
application is hung, which makes napari seem totally broken.

Napari is very extensible and customizable and users can create what amounts to
custom applications built on top of napari. So when the napari UI is blocked
it's not just "image viewing" that's blocked, their whole application becomes
unusable.

 For all of these reasons we'd like napari's GUI thread to never block.

# Framerate

Most screens refresh at 60Hz. To look and feel fully responsive a GUI
application should strive to draw at 60Hz as well. If 60Hz is not possible,
however, refreshing as fast as possible is important because the user experience
degrades rapidly as the refresh rate gets slower:

| Framerate | Milliseconds | User Experience |
| --------- | ------------ | --------------- |
| 60Hz      | 16.7         | Great           |
| 30Hz      | 33.3         | Good            |
| 20Hz      | 50           | Acceptable      |
| 10Hz      | 100          | Bad             |
| 5Hz       | 200          | Unusable        |

In addition to the average rate dropping there can be single slow frames, or
stretches of slow frames sometimes called stuttering. These should be minimized
as well since they will make napari seem glitchy or flakey even if the average framerate is decent.

# Array-like Interface

Napari renders data out of an "array like" interface, which is any object that
presents an interface compatible with `numpy` slicing and access. Using
`dask` or custom code it's possible the data does not live in memory and will be
paged in as chunks from disk as it's accessed. It's also possible the data does
not exist at all and it will be computed on-the-fly when it is accessed.

In those of these cases the array access is not a simple memory access. Instead
accessing the array triggers code that can more or less do anything. It could therefore take a long time to execute.

In #845 the array access lead to IO from disk or over the network. In #1320 the
array access leads to a Machine Learning (Torch) calculation.

In #1300 the problem is different. There the data is already entirely in memory,
but it's not chunked. So we transfer a single large array, 100's of MB, on to
the card and this is slow.

# Goals

The above analysis leads to these design goals for rendering:

1. Always break data into small "chunks" to send to the graphics card.
2. Never call `asarray` on user data from the GUI thread since we don't know
   what it will do or how long it will take.

# Chunks

**Chunks** is a deliberately vague term. A chunk is data used to render a
portion of the scene. Without chunks we are stuck rendering nothing or rendering
the entire scene. With chunks we can partially and progressively render the
scene using whatever chunks are available.

![render-frame](images/chunked-format.png)

The most common types of chunks are blocks of contiguous memory inside a chunked
file format like **Zarr** (on disk) and exposed by an API like *Dask*. If an
image is stored without chunks then reading a 2D rectangle would require
hundreds of small read operations from all over the file. With chunks reading
single span of data gets a single complete 2D rectangle.

For 3D images the chunks tend to be 3D blocks, but the idea is the same. With
Neuroglancer they commonly store the data in 64x64x64 voxel chunks. This is
useful because you can read the data in XY, XZ or YZ and it performs the same in
each orientation. It's also nice because you scroll through slices quickly since
you are reading up to 64 slices head of where you are.

In #1300 there are no chunks, the images were created in memory as one
monolithic thing, so we are going to have to break it into chunks in order to
send it to the graphics card incrementally. 

In #1320 the images are small so we are not chunked, but there are 3 image
layers, so we can consider the full layers to be chunks. In general we can get
creative with chunks, they can be spatial subdivisions or any other division we
want.

# Loading into RAM and VRAM

Data needs to be in VRAM before we draw it. This is a two step process: loading
it into RAM first them VRAM.

Loading into RAM must be done in a thread since we don't know how long it will
take. For example loading data over the internet or doing a complex calculation
to produce the data could both take really long time. We are going to use the
new `@thread_worker` interface for our thread pool.

Loading into VRAM is a different story because it must happen in the GUI thread, at least for now with OpenGL. Therefore we need to amortize the load over some number of frames. We will set a a budget, for example 5 milliseconds. Each frame can spend that much time loading data into VRAM, it will spend the rest of the frame drawing as normal.

![render-frame](images/paging-chunks.png)


