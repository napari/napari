import fsspec, zarr
import dask.array as da
import napari

# access the root of the n5 container
group = zarr.open(zarr.N5FSStore('s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5', anon=True))

# s0 (highest resolution) through s5 (lowest resolution) are available
data = []
for i in range(0, 5):
    zarr_array = group[f'em/fibsem-uint16/s{i}']
    data.append(da.from_zarr(zarr_array, chunks=zarr_array.chunks))

# This order presents a better visualization, but seems to break simple async.
#viewer = napari.view_image(data, order=(1, 0, 2), contrast_limits=(18000, 40000), multiscale=True)
viewer = napari.view_image(data, contrast_limits=(18000, 40000), multiscale=True)

if __name__ == '__main__':
    napari.run()

