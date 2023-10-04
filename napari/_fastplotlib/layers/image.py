from fastplotlib import ImageGraphic
from napari.layers import Image


class FastplotlibImageLayer:
    def __init__(
            self,
            napari_layer: Image
    ) -> None:
        """Create a fastplotlib ImageGraphic from a napari Image layer

        Parameters
        ----------
        napari_layer : napari.layers.Image
            Image layer to connect to fastplotlib ImageGraphic

        Attributes
        ----------
        napari_layer : napari.layers.Image
            Image layer to connect to fastplotlib ImageGraphic
        image_graphic : fastplotlib.ImageGraphic
            ImageGraphic generated from napari Image layer
        """

        self.napari_layer = napari_layer

        self.image_graphic = None

        self._on_data_change()

        self.napari_layer.events.colormap.connect(self._on_colormap_change)
        self.napari_layer.events.contrast_limits.connect(self._on_contrast_limits_change)

    def _on_data_change(self):
        # create image graphic
        if self.image_graphic is None:
            self.image_graphic = ImageGraphic(
                data=self.napari_layer._data_view,
                name=self.napari_layer.name,
                cmap=self.napari_layer.colormap.name,
                vmin=self.napari_layer.contrast_limits[0],
                vmax=self.napari_layer.contrast_limits[1]
            )
        else:
            if self.napari_layer._data_view.shape == self.image_graphic.data().shape:
                self.image_graphic.data = self.napari_layer._data_view
            else:
                raise ValueError(f"Current data shape: {self.image_graphic.data().shape} does not equal"
                                 f"new data shape: {self.napari_layer._data_view}. Please ")

    def _on_colormap_change(self):
        self.image_graphic.cmap = self.napari_layer.colormap.name
        self.image_graphic.present = False
        self.image_graphic.present = True

    def _on_contrast_limits_change(self):
        self.image_graphic.cmap.vmin = self.napari_layer.contrast_limits[0]
        self.image_graphic.cmap.vmax = self.napari_layer.contrast_limits[1]
