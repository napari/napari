from napari._vispy.layers.image import VispyImageLayer


class VispyLabelsLayer(VispyImageLayer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, texture_format=None, **kwargs)

        self.layer.events.labels_update.connect(self._on_labels_update)

    def _on_labels_update(self, event):
        if not self.layer.loaded or event.updated_slice is None:
            return

        dims_displayed = self.layer._slice_input.displayed
        raw_displayed = self.layer._slice.image.raw
        ndims = len(dims_displayed)

        if self.node._texture.shape[:ndims] != raw_displayed.shape[:ndims]:
            self.layer.refresh()
            return

        updated_slice = event.updated_slice

        # Keep only the dimensions that correspond to the current view
        updated_slice = tuple(
            [updated_slice[index] for index in dims_displayed]
        )

        offset = [axis_slice.start for axis_slice in updated_slice]

        colors_sliced = self.layer._raw_to_displayed(
            raw_displayed, data_slice=updated_slice
        )

        self.node._texture.scale_and_set_data(
            colors_sliced, copy=False, offset=offset
        )
        self.node.update()
