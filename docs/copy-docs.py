"""
Copy docs from the napari repo into the napari.github.io repo
and update the table of contents.
By default, will assume that there is a folder named napari.github.io
in the same directory as the napari folder, if not a different copy
destination can be provided.

Read ORGANIZATION.md to learn more about how the documentation sources
are organized, and how everything comes together.

python -m copy-docs [dstdir]
"""

import copy
import os
import os.path as osp
import shutil
import sys
from fnmatch import fnmatch

import yaml

# path to copy and locations to copy to if different
TO_COPY = [
    'ORGANIZATION.md',
    'glossary.md',
    'developers',
    'community',
    'howtos',
    'release',
    'roadmaps',
    'images',
    osp.join('_templates', 'autosummary'),
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

DOC_EXTS = ['.md', '.rst', '.ipynb']

TOC_IGNORE = [
    'api/stable',
    'images',
    '_templates',
    'ORGANIZATION.md',
    'glossary.md',  # this document will still be at the destination ToC
    'guides/stable/_layer_events.md',
    'guides/stable/_viewer_events.md',
    'plugins/stable/_npe2_contributions.md',
    'plugins/stable/_npe2_manifest.md',
    'plugins/stable/_npe2_readers_guide.md',
    'plugins/stable/_npe2_widgets_guide.md',
    'plugins/stable/_npe2_writers_guide.md',
    'plugins/stable/_npe2_sample_data_guide.md',
    'plugins/stable/_layer_data_guide.md',
]


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

    Returns
    -------
    files : list of path-like
        Paths of the copied files.
    """
    if newpath is None:
        newpath = path

    src = osp.join(srcdir, path)
    dst = osp.join(dstdir, newpath)

    if exclude(src):  # skip this path
        return []

    print(f'copying {src} to {dst}')

    if osp.isfile(src):
        shutil.copyfile(src, dst)
        return [newpath]
    elif osp.isdir(src):
        if osp.exists(dst):  # if the destination directory exists, delete it
            shutil.rmtree(dst)

        os.mkdir(dst)

        files = []

        for fpath in os.listdir(src):  # recursively copy each child path
            p = osp.join(path, fpath)
            np = osp.join(newpath, fpath)
            files += copy_path(srcdir, dstdir, p, np, exclude=exclude)

        return files
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

    Returns
    -------
    files : list of path-like
        Paths of the copied files.
    """
    files = []

    for path in paths:
        if isinstance(path, tuple):
            files += copy_path(src, dst, path[0], path[1], exclude=exclude)
        else:
            files += copy_path(src, dst, path, exclude=exclude)

    return files


def update_toc(toc, paths, ignore=[]):
    """Update the table of contents according to the paths of all files copied over.

    Parameters
    ----------
    toc : JSON
        Table of contents according to the JupyterBook specification.
    paths : list of path-like
        Paths of the files copied over.
    ignore : list of path-like
        List of directories to ignore when updating the table of contents.

    Returns
    -------
    new_toc : JSON
        Updated table of contents.
    """
    new_toc = copy.deepcopy(toc)

    remaining_paths = []

    # remove all paths in ignore list and those with the wrong extension
    for path in paths:
        base, ext = osp.splitext(path)

        for prefix in ignore:  # check if path should be ignored
            if path.startswith(prefix):
                break
        else:  # not on the ignore list
            if ext in DOC_EXTS:  # relevant filetype
                remaining_paths.append(
                    base
                )  # the toc does not include extensions

    chapters = new_toc[1]['chapters']

    for chapter in chapters:
        if (
            'file' not in chapter
            or (index := chapter['file']) not in remaining_paths
        ):
            continue  # skip irrelevant chapters

        parent_dir = osp.dirname(index)
        remaining_paths.remove(index)

        sections = chapter['sections']
        files = [section['file'] for section in sections]

        # find and remove deleted files from toc
        j = 0
        for path in files:
            if path in remaining_paths:
                remaining_paths.remove(path)
                j += 1
            else:
                print(f'deleting {path} from toc')
                del sections[j]  # delete from toc

        new_files = filter(
            lambda path: path.startswith(parent_dir), remaining_paths
        )
        for path in new_files:
            print(f'adding {path} to toc')
            sections.append({'file': path})
            remaining_paths.remove(path)

    return new_toc


def main(args):
    dst = osp.join(
        osp.dirname(osp.dirname(osp.dirname(__file__))), 'napari.github.io'
    )

    try:
        dst = args[1]
    except IndexError:
        pass

    files = copy_paths(SRC, dst, TO_COPY, exclude=exclude_filter)
    toc_file = osp.join(dst, '_toc.yml')

    with open(toc_file) as f:
        toc = yaml.safe_load(f)

    if toc is None:
        print(f'toc file {toc_file} empty')
        return

    new_toc = update_toc(toc, files, TOC_IGNORE)

    with open(toc_file, 'w') as f:
        yaml.dump(new_toc, f)


if __name__ == '__main__':
    main(sys.argv)
