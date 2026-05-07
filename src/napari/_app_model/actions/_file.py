from app_model.types import (
    Action,
    SubmenuItem,
)

from napari._app_model.constants import MenuGroup, MenuId
from napari.components import LayerList, ViewerModel
from napari.layers import Layer, Points, Shapes

FILE_SUBMENUS = [
    (
        MenuId.MENUBAR_FILE,
        SubmenuItem(
            submenu=MenuId.FILE_NEW_LAYER,
            title='New Layer',
            group=MenuGroup.NAVIGATION,
            order=0,
        ),
    ),
    (
        MenuId.MENUBAR_FILE,
        SubmenuItem(
            submenu=MenuId.FILE_OPEN_WITH_PLUGIN,
            title='Open with Plugin',
            group=MenuGroup.OPEN,
            order=99,
        ),
    ),
    (
        MenuId.MENUBAR_FILE,
        SubmenuItem(
            submenu=MenuId.FILE_SAMPLES,
            title='Open Sample',
            group=MenuGroup.OPEN,
            order=100,
        ),
    ),
    (
        MenuId.MENUBAR_FILE,
        SubmenuItem(
            submenu=MenuId.FILE_IO_UTILITIES,
            title='IO Utilities',
            group=MenuGroup.UTIL,
            order=101,
        ),
    ),
    (
        MenuId.MENUBAR_FILE,
        SubmenuItem(
            submenu=MenuId.FILE_ACQUIRE,
            title='Acquire',
            group=MenuGroup.UTIL,
            order=101,
        ),
    ),
]


def add_new_points(viewer: 'ViewerModel') -> None:
    if not viewer.layers.selection:
        ndim = max(viewer.dims.ndim, 2)
        viewer.add_points(  # type: ignore[attr-defined]
            ndim=ndim,
            scale=(1,) * ndim,
            units=viewer.layers.extent.units,
        )
    else:
        extent = viewer.layers.get_extent(viewer.layers.selection)
        viewer.add_points(  # type: ignore[attr-defined]
            ndim=len(extent.step),
            scale=extent.step,
            units=viewer.layers.extent.units,
        )


def add_new_shapes(viewer: 'ViewerModel') -> None:
    if not viewer.layers.selection:
        ndim = max(viewer.dims.ndim, 2)
        viewer.add_shapes(  # type: ignore[attr-defined]
            ndim=ndim,
            scale=(1,) * ndim,
            units=viewer.layers.extent.units,
        )
    else:
        extent = viewer.layers.get_extent(viewer.layers.selection)
        viewer.add_shapes(  # type: ignore[attr-defined]
            ndim=len(extent.step),
            scale=extent.step,
            units=viewer.layers.extent.units,
        )


def _create_single_layer(
    source_layer: Layer,
    layer_class: type[Points] | type[Shapes],
    layer_name: str,
) -> Points | Shapes:
    """Create a single Points or Shapes layer from the given source layer."""
    return layer_class(
        name=layer_name,
        ndim=source_layer.ndim,
        scale=source_layer.scale,
        translate=source_layer.translate,
        rotate=source_layer.rotate,
        shear=source_layer.shear,
        units=source_layer.units,
        affine=source_layer.affine.affine_matrix,
    )


def get_layer_name(base_name: str, existing_names: set[str]) -> str:
    """Generate a unique layer name based on the base name and existing names."""
    if base_name not in existing_names:
        return base_name
    index = 1
    while f'{base_name} {index}' in existing_names:
        index += 1
    return f'{base_name} {index}'


def new_labels(viewer: ViewerModel) -> None:
    viewer._new_labels()


def _new_layer_from_active(
    layer_list: LayerList, layer_class: type[Points] | type[Shapes]
) -> Points | Shapes:
    """Create a new layer from the given layer list."""
    if layer_list.selection.active is None:
        raise ValueError('No active layer to create new layer from.')
    source_layer = layer_list.selection.active
    new_layer_name = get_layer_name(
        f'{source_layer.name} - {layer_class.__name__}',
        existing_names={layer.name for layer in layer_list},
    )
    return _create_single_layer(source_layer, layer_class, new_layer_name)


def new_points(viewer: ViewerModel) -> None:
    if viewer.layers.selection.active is not None:
        viewer.add_layer(_new_layer_from_active(viewer.layers, Points))
    else:
        add_new_points(viewer)


def new_shapes(viewer: ViewerModel) -> None:
    if viewer.layers.selection.active is not None:
        viewer.add_layer(_new_layer_from_active(viewer.layers, Shapes))
    else:
        add_new_shapes(viewer)


FILE_ACTIONS: list[Action] = [
    Action(
        id='napari.window.file.new_layer.new_labels',
        title='Labels',
        callback=new_labels,
        menus=[{'id': MenuId.FILE_NEW_LAYER, 'group': MenuGroup.NAVIGATION}],
    ),
    Action(
        id='napari.window.file.new_layer.new_points',
        title='Points',
        callback=new_points,
        menus=[{'id': MenuId.FILE_NEW_LAYER, 'group': MenuGroup.NAVIGATION}],
    ),
    Action(
        id='napari.window.file.new_layer.new_shapes',
        title='Shapes',
        callback=new_shapes,
        menus=[{'id': MenuId.FILE_NEW_LAYER, 'group': MenuGroup.NAVIGATION}],
    ),
]
