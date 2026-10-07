import os

import fsspec
import zarr
from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
from fsspec.utils import get_protocol
from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import FsspecStore


class _CompleteArrayStore(FsspecStore):
    """Read-only store for an array whose every chunk exists on the server."""

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        try:
            return await super().get(key, prototype, byte_range)
        except FileNotFoundError as e:
            # the cache reports a server error or a dropped connection the
            # same way as a missing file, so the two cannot be told apart here
            raise ConnectionError(
                f'Could not fetch {key!r} from {self.path}: the server '
                'returned an error or the file does not exist.'
            ) from e


def open_cached_zarr(
    url: str, cache_dir: str | os.PathLike[str]
) -> zarr.Array:
    """Open a remote zarr v3 array, keeping each chunk it reads in `cache_dir`.

    Chunks already in `cache_dir` are read from disk without contacting the
    server, and chunks that are never read are never downloaded. Every chunk of
    the array must exist on the server, because a chunk the server does not
    return raises instead of being filled with the array's fill value.

    .. versionadded:: 0.10.0

    Raises
    ------
    ConnectionError
        If the server does not return the array metadata or a chunk.
    """
    cached_fs = fsspec.filesystem(
        'simplecache',
        target_protocol=get_protocol(url),
        cache_storage=os.fspath(cache_dir),
    )
    store = _CompleteArrayStore(
        fs=AsyncFileSystemWrapper(cached_fs, asynchronous=True),
        path=url,
        read_only=True,
        allowed_exceptions=(),
    )
    return zarr.open_array(store, zarr_format=3, mode='r')
