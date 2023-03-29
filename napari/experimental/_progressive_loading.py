from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

from fibsem_tools import read_xarray
from cachey import Cache


# A ChunkCacheManager manages multiple chunk caches
class ChunkCacheManager:
    def __init__(self, cache_size=1e9, cost_cutoff=0):
        """
        cache_size, size of cache in bytes
        cost_cutoff, cutoff anything with cost_cutoff or less
        """
        self.c = Cache(cache_size, cost_cutoff)

    def put(self, container, dataset, chunk_slice, value, cost=1):
        """Associate value with key in the given container.
        Container might be a zarr/dataset, key is a chunk_slice, and
        value is the chunk itself.
        """
        k = self.get_container_key(container, dataset, chunk_slice)
        self.c.put(k, value, cost=cost)

    def get_container_key(self, container, dataset, slice_key):
        """Create a key from container, dataset, and chunk_slice

        Parameters
        ----------
        container : str
            A string representing a zarr container
        dataset : str
            A string representing a dataset inside a zarr
        chunk_slice : slice
            A ND slice for the chunk to grab

        """
        if type(slice_key) is tuple:
            slice_key = ",".join(
                [f"{st.start}:{st.stop}:{st.step}" for st in slice_key]
            )

        return f"{container}/{dataset}@({slice_key})"

    def get(self, container, dataset, chunk_slice):
        """Get a chunk associated with the container, dataset, and chunk_size

        Parameters
        ----------
        container : str
            A string represening a zarr container
        dataset : str
            A string representing a dataset inside the container
        chunk_slice : slice
            A ND slice for the chunk to grab

        """
        return self.c.get(
            self.get_container_key(container, dataset, chunk_slice)
        )

# Example data loaders

# TODO capture some sort of metadata about scale factors
def openorganelle_mouse_kidney_labels():
    large_image = {
        "container": "s3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.n5",
        "dataset": "labels/empanada-mito_seg",
        "scale_levels": 4,
        "scale_factors": [(1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8)],
    }
    large_image["arrays"] = [
        read_xarray(
            f"{large_image['container']}/{large_image['dataset']}/s{scale}/",
            storage_options={"anon": True},
        ).data
        for scale in range(large_image["scale_levels"])
    ]
    return large_image


def openorganelle_mouse_kidney_em():
    large_image = {
        "container": "s3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.n5",
        "dataset": "em/fibsem-uint8",
        "scale_levels": 5,
        "scale_factors": [
            (1, 1, 1),
            (2, 2, 2),
            (4, 4, 4),
            (8, 8, 8),
            (16, 16, 16),
        ],
    }
    large_image["arrays"] = [
        read_xarray(
            f"{large_image['container']}/{large_image['dataset']}/s{scale}/",
            storage_options={"anon": True},
        ).data
        for scale in range(large_image["scale_levels"])
    ]
    return large_image


# TODO this one needs testing, it is chunked over 5D
def idr0044A():
    large_image = {
        "container": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0044A/4007801.zarr",
        "dataset": "",
        "scale_levels": 5,
        "scale_factors": [
            (1, 1, 1),
            (1, 2, 2),
            (1, 4, 4),
            (1, 8, 8),
            (1, 16, 16),
        ],
    }
    large_image["arrays"] = [
        read_xarray(
            f"{large_image['container']}/{scale}/",
            #            storage_options={"anon": True},
        ).data.rechunk((1, 1, 128, 128, 128))
        # .data[362, 0, :, :, :].rechunk((512, 512, 512))
        for scale in range(large_image["scale_levels"])
    ]
    return large_image


def idr0075A():
    large_image = {
        "container": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/idr0075A/9528933.zarr",
        "dataset": "",
        "scale_levels": 4,
        "scale_factors": [(1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 8, 8)],
    }
    large_image["arrays"] = [
        read_xarray(
            f"{large_image['container']}/{scale}/",
            #            storage_options={"anon": True},
        ).data
        # .data[362, 0, :, :, :].rechunk((512, 512, 512))
        for scale in range(large_image["scale_levels"])
    ]
    # .rechunk((1, 1, 128, 128, 128))
    return large_image


def idr0051A():
    large_image = {
        "container": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/idr0051A/4007817.zarr",
        "dataset": "",
        "scale_levels": 3,
        "scale_factors": [(1, 1, 1), (1, 2, 2), (1, 4, 4)],
    }
    large_image["arrays"] = [
        read_xarray(
            f"{large_image['container']}/{scale}/",
            #            storage_options={"anon": True},
        ).data
        # .data[362, 0, :, :, :].rechunk((512, 512, 512))
        for scale in range(large_image["scale_levels"])
    ]
    # .rechunk((1, 1, 128, 128, 128))
    return large_image


def luethi_zenodo_7144919():
    import os
    import pooch

    # Downloaded from https://zenodo.org/record/7144919#.Y-OvqhPMI0R
    # TODO use pooch to fetch from zenodo
    # zip_path = pooch.retrieve(
    #     url="https://zenodo.org/record/7144919#.Y-OvqhPMI0R",
    #     known_hash=None,# Update hash
    # )
    dest_dir = pooch.retrieve(
        url="https://zenodo.org/record/7144919/files/20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip?download=1",
        known_hash="e6773fc97dcf3689e2f42e6504e0d4f4d0845c329dfbdfe92f61c2f3f1a4d55d",
        processor=pooch.Unzip(),
    )
    local_container = os.path.split(dest_dir[0])[0]
    print(local_container)
    store = parse_url(local_container, mode="r").store
    reader = Reader(parse_url(local_container))
    nodes = list(reader())
    image_node = nodes[0]
    dask_data = image_node.data

    large_image = {
        "container": local_container,
        "dataset": "B/03/0",
        "scale_levels": 5,
        "scale_factors": [
            (1, 0.1625, 0.1625),
            (1, 0.325, 0.325),
            (1, 0.65, 0.65),
            (1, 1.3, 1.3),
            (1, 2.6, 2.6),
        ],
        "chunk_size": (1, 10, 256, 256)
    }
    large_image["arrays"] = []
    for scale in range(large_image["scale_levels"]):
        array = dask_data[scale]

        # TODO extract scale_factors now

        # large_image["arrays"].append(result.data.rechunk((3, 10, 256, 256)))
        large_image["arrays"].append(
            array.rechunk((1, 10, 256, 256)).squeeze()
            # result.data[2, :, :, :].rechunk((10, 256, 256)).squeeze()
        )
    return large_image
