# Napari benchmarking with airspeed velocity (asv)

These are benchmarks to be run with airspeed velocity 
([asv](https://asv.readthedocs.io/en/stable/)). They are not distributed with
installs.

## Example commands

Run all the benchmarks:

`asv run`

Do a "quick" run in the current environment, where each benchmark function is run only once:

`asv run --python=same -q`

To run a single benchmark (Vectors3DSuite.time_refresh) with the environment you are currently in:

`asv dev --bench Vectors3DSuite.time_refresh`

To compare benchmarks across branches, run using conda environments (instead of virtualenv), and limit to the `Labels2DSuite` benchmarks:

`asv continuous main fix_benchmark_ci -q --environment conda --bench Labels2DSuite`
