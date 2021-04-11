"""Copy docs from the napari repo into the napari.github.io repo

python -m copy-docs.py [dstdir]
"""

import os
import os.path as osp
import shutil
import sys
from fnmatch import fnmatch

# path to copy and locations to copy to if different
TO_COPY = [
    'developers',
    'community',
    'release',
    'roadmaps',
    'images',
    '_templates',
    *[
        (dire, osp.join(dire, 'stable'))
        for dire in ('api', 'guides', 'plugins')
    ],
]

# paths to ignore
IGNORE = [
    osp.join('images', 'logo.png'),
]

SRC = osp.dirname(__file__)


def exclude_filter(path):
    """Exclude files in the ignore list and duplicated files."""
    for ignore in IGNORE:
        if fnmatch(path, osp.join(SRC, ignore)):  # in ignore list
            return True
    else:
        if osp.isdir(path) or osp.splitext(path)[1] != '.md':
            return False
        with open(path) as f:
            firstline = f.readline()
        return firstline.startswith('```{include}')  # duplicate file


def copy_path(srcdir, dstdir, path, newpath=None, *, exclude=None):
    """Copy a path from the source directory to the destination directory,
    with the given path relative to the directory roots.

    Parameters
    ----------
    srcdir : path-like
        Source directory root to copy from.
    dstdir : path-like
        Destination directory root to copy to.
    path : path-like
        Path relative to the `srcdir` of the path to copy from.
    newpath : path-like, optional
        Path relative to the `dstdir` of the path to copy to.
        If not provided, will default to the value of `path`.
    exclude : function(path-like) -> bool, keyword-only, optional
        Conditional function on whether to exclude the given path.
    """
    if newpath is None:
        newpath = path

    src = osp.join(srcdir, path)
    dst = osp.join(dstdir, newpath)

    if exclude(src):  # skip this path
        return

    print(f'copying {src} to {dst}')

    if osp.isfile(src):
        shutil.copyfile(src, dst)
    elif osp.isdir(src):
        if osp.exists(dst):  # if the destination directory exists, delete it
            shutil.rmtree(dst)

        os.mkdir(dst)

        for fpath in os.listdir(src):  # recursively copy each child path
            copy_path(src, dst, fpath, exclude=exclude)
    else:
        raise RuntimeError(f'unknown path type {src}')


def copy_paths(src, dst, paths, *, exclude=None):
    """Copy files/directories given a list of their paths from
    the source directory to the destination directory.

    Parameters
    ----------
    src : path-like
        Source directory to copy from.
    dst : path-like
        Destination directory to copy to.
    paths : list of (path-like or 2-tuple of path-like)
        Paths of the files/directories to copy relative to the source directory.
        Pairs of paths in the list signify that the path to copy to is different
        than the path copied from.
    exclude : function(path-like) -> bool, keyword-only, optional
        Conditional function on whether to exclude the given path.
    """
    for path in paths:
        if isinstance(path, tuple):
            copy_path(src, dst, path[0], path[1], exclude=exclude)
        else:
            copy_path(src, dst, path, exclude=exclude)


def main(args):
    dst = osp.join(
        osp.dirname(osp.dirname(osp.dirname(__file__))), 'napari.github.io'
    )

    try:
        dst = args[1]
    except IndexError:
        pass

    copy_paths(SRC, dst, TO_COPY, exclude=exclude_filter)


if __name__ == '__main__':
    main(sys.argv)
