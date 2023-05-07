from napari._vispy.layers.image import VispyImageLayer


class VispyLabelsLayer(VispyImageLayer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, texture_format=None, **kwargs)

        self.layer.events.labels_update.connect(self._on_labels_update)

    def _on_labels_update(self, event):
        if not self.layer.loaded or event.updated_slice is None:
            return

        updated_slice = event.updated_slice
        offset = [start_index for start_index, _ in updated_slice]
        updated_slice = tuple(
            [slice(start, stop) for start, stop in updated_slice]
        )

        updated_raw = self.layer._raw_to_displayed(
            self.layer.data, data_slice=updated_slice
        )

        self.node._texture.scale_and_set_data(
            updated_raw, copy=False, offset=offset
        )
        self.node.update()
