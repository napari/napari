# flake8: noqa
from typing import Any, Dict, List, Sequence, Union

import napari

class ViewerModel(
    napari.utils.key_bindings.KeymapProvider,
    napari.utils.mouse_bindings.MousemapProvider,
    napari.utils.events.evented_model.EventedModel,
):
    def _add_layer_from_data(
        self,
        data,
        meta: Dict[str, Any] = None,
        layer_type: Union[str, None] = None,
    ) -> Union[
        napari.layers.base.base.Layer, List[napari.layers.base.base.Layer]
    ]: ...
    def _add_layers_with_plugins(
        self,
        path_or_paths: Union[str, Sequence[str]],
        kwargs: Union[dict, None] = None,
        plugin: Union[str, None] = None,
        layer_type: Union[str, None] = None,
    ) -> List[napari.layers.base.base.Layer]: ...
    def _new_labels(self): ...
    def _on_add_layer(self, event): ...
    def _on_cursor_position_change(self, event): ...
    def _on_grid_change(self, event): ...
    def _on_layers_change(self, event): ...
    def _on_remove_layer(self, event): ...
    def _subplot(self, layer, position, extent): ...
    def _toggle_theme(self): ...
    def _update_active_layer(self, event): ...
    def _update_cursor(self, event): ...
    def _update_cursor_size(self, event): ...
    def _update_interactive(self, event): ...
    def _update_layers(self, event=None, layers=None): ...
    def _valid_theme(v): ...
    def add_image(
        self,
        data=None,
        *,
        channel_axis=None,
        rgb=None,
        colormap=None,
        contrast_limits=None,
        gamma=1,
        interpolation="nearest",
        rendering="mip",
        iso_threshold=0.5,
        attenuation=0.05,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=1,
        blending=None,
        visible=True,
        multiscale=None
    ) -> Union[
        napari.layers.image.image.Image, List[napari.layers.image.image.Image]
    ]: ...
    def add_labels(
        self,
        data,
        *,
        num_colors=50,
        properties=None,
        color=None,
        seed=0.5,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=0.7,
        blending="translucent",
        visible=True,
        multiscale=None
    ): ...
    def add_layer(
        self, layer: napari.layers.base.base.Layer
    ) -> napari.layers.base.base.Layer: ...
    def add_points(
        self,
        data=None,
        *,
        ndim=None,
        properties=None,
        text=None,
        symbol="o",
        size=10,
        edge_width=1,
        edge_color="black",
        edge_color_cycle=None,
        edge_colormap="viridis",
        edge_contrast_limits=None,
        face_color="white",
        face_color_cycle=None,
        face_colormap="viridis",
        face_contrast_limits=None,
        n_dimensional=False,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=1,
        blending="translucent",
        visible=True
    ): ...
    def add_shapes(
        self,
        data=None,
        *,
        ndim=None,
        properties=None,
        text=None,
        shape_type="rectangle",
        edge_width=1,
        edge_color="black",
        edge_color_cycle=None,
        edge_colormap="viridis",
        edge_contrast_limits=None,
        face_color="white",
        face_color_cycle=None,
        face_colormap="viridis",
        face_contrast_limits=None,
        z_index=0,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=0.7,
        blending="translucent",
        visible=True
    ): ...
    def add_surface(
        self,
        data,
        *,
        colormap="gray",
        contrast_limits=None,
        gamma=1,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=1,
        blending="translucent",
        visible=True
    ): ...
    def add_tracks(
        self,
        data,
        *,
        properties=None,
        graph=None,
        tail_width=2,
        tail_length=30,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=1,
        blending="additive",
        visible=True,
        colormap="turbo",
        color_by="track_id",
        colormaps_dict=None
    ): ...
    def add_vectors(
        self,
        data,
        *,
        properties=None,
        edge_width=1,
        edge_color="red",
        edge_color_cycle=None,
        edge_colormap="viridis",
        edge_contrast_limits=None,
        length=1,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=0.7,
        blending="translucent",
        visible=True
    ): ...
    def open(
        self,
        path: Union[str, Sequence[str]],
        *,
        stack: bool = False,
        plugin: Union[str, None] = None,
        layer_type: Union[str, None] = None,
        **kwargs
    ) -> List[napari.layers.base.base.Layer]: ...
    def reset_view(self, event=None): ...
