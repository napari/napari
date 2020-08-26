"""ChunkLoader related utilities.
"""


def get_data_id(layer) -> int:
    """Return the data_id to use for this layer.

    Parameters
    ----------
    layer
        The layer to get the data_id from.

    Notes
    -----
    We use data_id rather than layer_id in case someone changes the data
    out from under a layer.
    """
    data = layer.data
    if isinstance(data, list):
        # Assert for now, but shapes layers do have an empty list.
        assert data

        # Just use the ID from the 0'th layer.
        return id(data[0])

    # Not a list so just use its id.
    return id(data)
