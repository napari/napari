# Utilties for napari performance monitoring

This directory contains configs and tools associated with [performance monitoring as described on napari.org](https://napari.org/stable/howtos/perfmon.html?highlight=perfmon).

Storing these in the repo makes it easier to reproduce monitoring experiments and results by standardizing configurations and tooling.

Napari developers would be encouraged to add configurations to focus on specific areas of concern, e.g. slicing.
Users can then be encouraged to use this tool to help developers better understand napari's performance in the wild.

## Usage

From the root of the napari repo:
```shell
% python tools/perfmon/run.py CONFIG NAPARI_ARGS 
```

For example, to measure slicing performance on an image:
```shell
% python tools/perfmon/run.py slicing docs/images/multichannel_cells.png
```