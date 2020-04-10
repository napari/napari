"""Patching dask.array.optimization.optimize

without this patch everytime we index a single plane from a
delayed dask array of 3D tiff stacks, it re-reads the whole file.
Removing the call to dask.optimization.fuse fixes it.
"""
import sys

import dask.array.optimization
from dask.array.core import getter_inline
from dask.blockwise import fuse_roots, optimize_blockwise
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import cull, inline_functions
from dask.utils import ensure_dict


def uncache(exclude):
    """Remove package modules from cache except excluded ones.
    On next import they will be reloaded.

    Args:
        exclude (iter<str>): Sequence of module paths.
    """
    pkgs = []
    for mod in exclude:
        pkg = mod.split('.', 1)[0]
        pkgs.append(pkg)

    to_uncache = []
    for mod in sys.modules:
        if mod in exclude:
            continue

        if mod in pkgs:
            to_uncache.append(mod)
            continue

        for pkg in pkgs:
            if mod.startswith(pkg + '.'):
                to_uncache.append(mod)
                break

    for mod in to_uncache:
        del sys.modules[mod]


def optimize(
    dsk,
    keys,
    fuse_keys=None,
    fast_functions=None,
    inline_functions_fast_functions=(getter_inline,),
    rename_fused_keys=True,
    **kwargs,
):
    """ Optimize dask for array computation

    1.  Cull tasks not necessary to evaluate keys
    2.  Remove full slicing, e.g. x[:]
    3.  Inline fast functions like getitem and np.transpose
    """
    keys = list(flatten(keys))

    # High level stage optimization
    if isinstance(dsk, HighLevelGraph):
        dsk = optimize_blockwise(dsk, keys=keys)
        dsk = fuse_roots(dsk, keys=keys)

    # Low level task optimizations
    dsk = ensure_dict(dsk)
    if fast_functions is not None:
        inline_functions_fast_functions = fast_functions

    dsk2, dependencies = cull(dsk, keys)
    # this part causes the extra reads for 3D stacks
    # hold = dask.array.optimization.hold_keys(dsk2, dependencies)
    # dsk3, dependencies = fuse(
    #     dsk2,
    #     hold + keys + (fuse_keys or []),
    #     dependencies,
    #     rename_keys=rename_fused_keys,
    # )
    if inline_functions_fast_functions:
        dsk4 = inline_functions(
            dsk2,
            keys,
            dependencies=dependencies,
            fast_functions=inline_functions_fast_functions,
        )
    else:
        dsk4 = dsk2
    dsk5 = dask.array.optimization.optimize_slices(dsk4)

    return dsk5


dask.array.optimization.optimize = optimize

uncache(['dask.array.optimization'])
