import fsspec, zarr
import dask.array as da
import napari

group = zarr.open(zarr.N5FSStore('s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5', anon=True)) # access the root of the n5 container

# s0 (highest resolution) through s5 (lowest resolution) are available,
# so s3 is a suitable choice in the middle
zarr_array = group['em/fibsem-uint16/s3']
data = da.from_zarr(zarr_array, chunks=zarr_array.chunks)

# This order presents a better visualization, but seems to break simple async.
#viewer = napari.view_image(data, order=(1, 0, 2), contrast_limits=(18000, 40000))
viewer = napari.view_image(data, contrast_limits=(18000, 40000))
napari.run()

