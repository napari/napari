(napari-benchmarks)=
# Benchmarks

While not mandatory for most pull requests, we ask that performance related
PRs include a benchmark in order to clearly depict the use-case that is being
optimized for.

In this section we will review how to setup the benchmarks,
and three commands ``asv dev``, ``asv run`` and ``asv continuous``.

## Prerequisites

Begin by installing [airspeed velocity](https://asv.readthedocs.io/en/stable/)
in your development environment. Prior to installation, be sure to activate your
development environment, then if using ``venv`` you may install the requirement with:

```bash
source napari-dev/bin/activate
pip install asv
```

If you are using conda, then the command:

```bash
conda activate napari-dev
conda install asv
```

is more appropriate. Once installed, it is useful to run the command:

```bash
asv machine
```

To let airspeed velocity know more information about your machine.

## Writing a benchmark

To write  benchmark, add a file in the ``napari/_benchmarks`` directory which
contains a class with one ``setup`` method and at least one method prefixed
with ``time_``.

The ``time_`` method should only contain code you wish to benchmark.
Therefore it is useful to move everything that prepares the benchmark scenario
into the ``setup`` method. This function is called before calling a ``time_``
method and its execution time is not factored into the benchmarks.

Take for example the ``ViewImageSuite`` benchmark:

```python
import numpy as np
import napari
from qtpy.QtWidgets import QApplication


class ViewImageSuite:
    """Benchmarks for viewing images in the viewer."""

    def setup(self):
        app = QApplication.instance() or QApplication([])
        np.random.seed(0)
        self.data = np.random.random((512, 512))
        self.viewer = None

    def teardown(self):
        self.viewer.window.close()

    def time_view_image(self):
        """Time to view an image."""
        self.viewer = napari.view_image(self.data)
```

Here, the creation of the image is completed in the ``setup`` method, and not
included in the reported time of the benchmark.

It is also possible to benchmark features such as peak memory usage. To learn
more about the features of `asv`, please refer to the official
[airspeed velocity documentation](http://asv.readthedocs.io/en/latest/writing_benchmarks.html).

## Testing the benchmarks locally

Prior to running the true benchmark, it is often worthwhile to test that the
code is free of typos. To do so, you may use the command:

```bash
asv dev -b ViewImageSuite
```

Where the ``ViewImageSuite`` above will be run once in your current environment
to test that everything is in order.

## Running your benchmark

The command above is fast, but doesn't test the performance of the code
adequately. To do that you may want to run the benchmark in your current
environment to see the performance of your change as you are developing new
features. The command ``asv run -E existing`` will specify that you wish to run
the benchmark in your existing environment. This will save a significant amount
of time since building napari can be a time consuming task:

```bash
asv run -E existing -b ViewImageSuite
```

## Comparing results to main

Often, the goal of a PR is to compare the results of the modifications in terms
of speed to a snapshot of the code that is in the main branch of the
``napari`` repository. The command ``asv continuous`` is of help here:

```bash
asv continuous main your-current-branch -b ViewImageSuite
```

This call will build out the environments specified in the ``asv.conf.json``
file and compare the performance of the benchmark between your current commit
and the code in the main branch.

The output may look something like:

```bash
$ asv continuous main your-current-branch -b ViewImageSuite
· Creating environments
· Discovering benchmarks
·· Uninstalling from conda-py3.7-cython-numpy1.15-scipy
·· Installing 544c0fe3 <benchmark_docs> into conda-py3.7-cython-numpy1.15-scipy.
· Running 4 total benchmarks (2 commits * 2 environments * 1 benchmarks)
[  0.00%] · For napari commit 37c764cb <benchmark_docs~1> (round 1/2):
[...]
[100.00%] ··· ...Image.ViewImageSuite.time_view_image           33.2±2ms

BENCHMARKS NOT SIGNIFICANTLY CHANGED.
```

In this case, the differences between HEAD on your-current-branch and main are not significant
enough for airspeed velocity to report.

## Profiling

The airspeed velocity tool also supports code profiling using [`cProfile`](https://docs.python.org/3/library/profile.html#module-cProfile). For detailed instructions on how to use the profiling functionality see the
[asv profiling documentation](https://asv.readthedocs.io/en/stable/using.html#running-a-benchmark-in-the-profiler).

To profile a particular benchmark in napari you can run

```bash
asv profile benchmark_qt_viewer.QtViewerSuite.time_create_viewer -g snakeviz --python=same
```

where `benchmark_qt_viewer` is the file name, `QtViewerSuite` is the test suite class name,
and `time_create_viewer` is the test method.

To profile a particular parameterized benchmark you can run

```bash
asv profile "benchmark_image_layer.Image2DSuite.time_create_layer\(512\)" -g snakeviz --python=same
```

where `benchmark_image_layer` is the file name, `Image2DSuite` is the test suite class name,
and `time_to_create_layer` is the test method and `512` is a valid parameter input to the test method.

Note that we in both these cases we have sent the output of the profiling to [snakeviz](http://jiffyclub.github.io/snakeviz/)
which you can pip install with

```bash
pip install snakeviz
```

and we use `--python=same` to profile against our current python environment.

## Running benchmarks on CI

Benchmarking on CI has two main parts - the Benchmark Action and the Benchmark Reporting Action. 

### The Benchmark Action

The benchmarks are set to run:
* On a schedule: once a week on Sunday
* On PRs with the `run-benchmark` label
* On workflow dispatch (manual trigger)

If the benchmarks fail during the scheduled run, an issue is opened in the repo to flag the occurance. 
If an issue has already been opened, it will add to the existing issue. 

The contender SHA is Github PR merge commit - a fake commit not available to users
Every time you want the benchmark ci to run in PR, you'll need to remove and re-add the `run-benchmark` label. 


### Benchmark Reporting Action

The benchmark Reporting Action will only run after the succesful completion of the Benchmark Action 
(regardless of comparison failures).
