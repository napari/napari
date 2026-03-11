from __future__ import annotations

import warnings
from collections.abc import Sequence
from logging import getLogger
from typing import TYPE_CHECKING, Any

from napari.layers import Layer
from napari.plugins import _npe2
from napari.types import LayerData, PathLike
from napari.utils.translations import trans

logger = getLogger(__name__)
if TYPE_CHECKING:
    from npe2.manifest.contributions import WriterContribution


def read_data_with_plugins(
    paths: Sequence[PathLike],
    plugin: str | None = None,
    stack: bool = False,
) -> tuple[list[LayerData] | None, str | None]:
    """Iterate reader hooks and return first non-None LayerData or None.

    This function returns as soon as the path has been read successfully,
    while raising any errors occurring during the reading process, and
    providing useful error messages.

    Parameters
    ----------
    paths : str, or list of string
        The of path (file, directory, url) to open
    plugin : str, optional
        Name of a plugin to use.  If provided, will force ``path`` to be read
        with the specified ``plugin``.  If the requested plugin cannot read
        ``path``, an exception will be raised.
    stack : bool
        See `Viewer.open`

    Returns
    -------
    LayerData : list of tuples, or None
        LayerData that can be passed to :func:`Viewer._add_layer_from_data()
        <napari.components.viewer_model.ViewerModel._add_layer_from_data>`.
        ``LayerData`` is a list tuples, where each tuple is one of
        ``(data,)``, ``(data, meta)``, or ``(data, meta, layer_type)`` .

        If no reader plugins were found (or they all failed), returns ``None``
    reader : str, optional
        The name of the reader plugin that was used, or None if no reader was found.
    """
    if plugin == 'builtins':
        warnings.warn(
            trans._(
                'The "builtins" plugin name is deprecated and will not work in a future version. Please use "napari" instead.',
                deferred=True,
            ),
        )
        plugin = 'napari'

    assert isinstance(paths, list)
    if not stack:
        assert len(paths) == 1

    res = _npe2.read(paths, plugin, stack=stack)
    if res is not None:
        ld_, reader = res
        return [] if _is_null_layer_sentinel(ld_) else list(ld_), reader

    return [], None


def save_layers(
    path: str,
    layers: list[Layer],
    *,
    plugin: str | None = None,
    _writer: WriterContribution | None = None,
) -> list[str]:
    """Write list of layers or individual layer to a path using writer plugins.

    If ``plugin`` is provided, will attempt to use a writer declared by that
    plugin, assuming it is compatible with the given layer(s) and ``path``.
    Otherwise, will use first compatible writer.

    Parameters
    ----------
    path : str
        A filepath, directory, or URL to write.
    layers : List[layers.Layer]
        Non-empty List of layers to be saved. Warns when the list
        of layers is empty.
    plugin : str, optional
        Name of the plugin to use for saving. If None then the first compatible
        writer will be used.

    Returns
    -------
    list of str
        File paths of any files that were written.
    """
    writer_name = ''
    if layers:
        written, writer_name = _write_layers_with_plugins(
            path, layers, plugin_name=plugin, _writer=_writer
        )
    else:
        warnings.warn(trans._('No layers to write.'))
        return []

    # If written is empty, something went wrong.
    # Generate a warning to tell the user what it was.
    if not written:
        if writer_name:
            warnings.warn(
                trans._(
                    "Plugin '{name}' tried to save layers but did not return any written paths.",
                    deferred=True,
                    name=writer_name,
                )
            )
        elif plugin:
            warnings.warn(
                trans._(
                    "Given plugin '{name}' is not a valid writer for {path}.",
                    deferred=True,
                    name=plugin,
                    path=path,
                )
            )
        else:
            warnings.warn(
                trans._(
                    'No data written! A plugin could not be found to write these {length} layers to {path}.',
                    deferred=True,
                    length=len(layers),
                    path=path,
                )
            )

    return written


def _is_null_layer_sentinel(layer_data: Any) -> bool:
    """Checks if the layer data returned from a reader function indicates an
    empty file. The sentinel value used for this is ``[(None,)]``.

    Parameters
    ----------
    layer_data : LayerData
        The layer data returned from a reader function to check

    Returns
    -------
    bool
        True, if the layer_data indicates an empty file, False otherwise
    """
    return (
        isinstance(layer_data, list)
        and len(layer_data) == 1
        and isinstance(layer_data[0], tuple)
        and len(layer_data[0]) == 1
        and layer_data[0][0] is None
    )


def _write_layers_with_plugins(
    path: str,
    layers: list[Layer],
    *,
    plugin_name: str | None = None,
    _writer: WriterContribution | None = None,
) -> tuple[list[str], str]:
    """Write data from multiple layers data with a plugin.

    Calls out to _npe2.write_layers which will get the first
    writer compatible with the given layer combination and specified file
    extension in ``path``, and attempt to write the file(s).

    Parameters
    ----------
    path : str
        The path (file, directory, url) to write.
    layers : List of napari.layers.Layer
        List of napari layers to write.
    plugin_name : str, optional
        If provided, force the plugin manager to use this plugin's writer.
        If none is available, or if it is
        incapable of handling the layers, this function will fail.

    Returns
    -------
    (written paths, writer name) as Tuple[List[str],str]

    written paths: List[str]
        Empty list when no plugin was found, otherwise a list of file paths,
        if any, that were written.
    writer name: str
        Name of the plugin selected to write the data.
    """
    written_paths, writer_name = _npe2.write_layers(
        path, layers, plugin_name, _writer
    )
    if written_paths or writer_name:
        return (written_paths, writer_name)
    return ([], '')
