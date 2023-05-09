from napari._vispy.layers.image import VispyImageLayer


class VispyLabelsLayer(VispyImageLayer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, texture_format=None, **kwargs)

        self.layer.events.labels_update.connect(self._on_partial_labels_update)

    def _on_partial_labels_update(self, event):
        if not self.layer.loaded:
            return

        raw_displayed = self.layer._slice.image.raw
        ndims = len(event.offset)

        if self.node._texture.shape[:ndims] != raw_displayed.shape[:ndims]:
            self.layer.refresh()
            return

        self.node._texture.scale_and_set_data(
            event.data, copy=False, offset=event.offset
        )
        self.node.update()
