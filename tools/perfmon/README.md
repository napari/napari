# Utilties for napari performance monitoring

This directory contains configs and tools associated with [performance monitoring as described on napari.org](https://napari.org/stable/howtos/perfmon.html?highlight=perfmon).

Storing these in the repo makes it easier to reproduce monitoring experiments and results by standardizing configurations and tooling.

Napari developers would be encouraged to add configurations to focus on specific areas of concern, e.g. slicing.
Users can then be encouraged to use this tool to help developers better understand napari's performance in the wild.

## Usage

From the root of the napari repo:

```shell
python tools/perfmon/run.py CONFIG EXAMPLE_SCRIPT
```

To take a specific example, let's say that we want to monitor `Layer.refresh`
while interacting with a multi-scale image in napari.

First, we would call the run command with the slicing config and one of the
built-in example scripts:

```shell
python tools/perfmon/run.py slicing examples/add_multiscale_image.py
```

After interacting with napari then quitting the application either through
the application menu or keyboard shortcut, a traces JSON file should be
output to the slicing subdirectory:

```shell
cat tools/perfmon/slicing/traces-latest.json
```

You can then plot the distribution of the `Layer.refresh` callable defined
in the slicing config:

```shell
python tools/perfmon/plot_callable.py slicing Layer.refresh
```

Next, you might want to switch to a branch, repeat a similar interaction
with the same configuration to measure a potential improvement to napari:

```shell
python tools/perfmon/run.py slicing examples/add_multiscale_image.py --output=test
```

By specifying the `output` argument, the trace JSON file is written to a
different location to avoid overwriting the first file:

```shell
cat tools/perfmon/slicing/traces-test.json
```

We can then generate a comparison of the two runs to understand if there
was an improvement:

```shell
python tools/perfmon/compare_callable.py slicing Layer.refresh latest test
```
