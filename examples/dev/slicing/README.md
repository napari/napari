# Slicing examples

The examples in this directory are for developers to test various aspects
of layer slicing. These are primarily designed to aide in the async effort ([NAP 4](../../../docs/naps/4-async-slicing.md)).

## Examples

### Examples of desirable behavior

These are a set of examples which are easy and non-frustrating to interact in napari
without async support. We want to ensure that these examples continue to be peformant.

* ebi_empiar_3D_with_labels.py [EMPIAR-10982](https://www.ebi.ac.uk/empiar/EMPIAR-10982/)
  * Real-world image & labels data (downloaded locally)
* points_example_smlm.py
  * Real-world points data (downloaded locally)

Additional examples from the main napari examples:
* add_multiscale_image.py
  * Access to in-memory multi-scale data

### Examples of undesirable behavior

These are a set of examples which currently cause undesirable behavior in napari, typically
resulting in non-responsive user interface due to synchornous slicing on large or remote data.

* random_shapes.py
  * A large number of shapes to stress slicing on a shapes layer
* random_points.py
  * A large number of random points to stress slicing on a points layer
* janelia_s3_n5_multiscale.py
  * Multi-scale remote image data in zarr format

## Performance monitoring

The [perfmon](../../../tools/perfmon/README.md) tooling can be used to monitor the data
access performance on these examples.