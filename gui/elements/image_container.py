from vispy import scene
from ..visuals.napari_image import NapariImage


# get available interpolation methods
interpolation_method_names = scene.visuals.Image(None).interpolation_functions
interpolation_method_names = list(interpolation_method_names)
interpolation_method_names.sort()
interpolation_method_names.remove('sinc')  # does not work well on my machine

# print(interpolation_method_names)
index_to_name = interpolation_method_names.__getitem__
name_to_index = interpolation_method_names.index


class ImageContainer:
    """

    Parameters
    ----------
    view : vispy.scene.widgets.ViewBox
    """
    def __init__(self, image, view, update_func):
        self.image = image
        self.view = view
        self.update = update_func

        self.image_visual = NapariImage(image, parent=view.scene,
                                        method='auto')

        self._brightness = 1
        self._interpolation_index = 0

        self.interpolation = 'nearest'

    def set_image(self, image, dimx=0, dimy=1):
        """Sets the image given the data.

        Parameters
        ----------
        image : array
            Image data to update with.
        dimx : int, optional
            Ordinal axis considered as the x-axis.
        dimy : int, optional
            Ordinal axis considered as the y-axis.
        """
        # TODO: use dimx, dimy for something
        self.image = image

        self.image_visual.set_data(image)
        self.view.camera.set_range()

    @property
    def interpolation(self):
        """string: Equipped interpolation method's name.
        """
        return index_to_name(self.interpolation_index)

    @interpolation.setter
    def interpolation(self, interpolation):
        self.interpolation_index = name_to_index(interpolation)

    @property
    def interpolation_index(self):
        """int: Index of the current interpolation method equipped.
        """
        return self._interpolation_index

    @interpolation_index.setter
    def interpolation_index(self, interpolation_index):
        intp_index = interpolation_index % len(interpolation_method_names)
        self._interpolation_index = intp_index
        self.image_visual.interpolation = index_to_name(intp_index)
        # print(self.image_visual.interpolation)
        self.update()

    @property
    def brightness(self):
        """float: Image brightness.
        """
        return self._brightness

    @brightness.setter
    def brightness(self, brightness):
        # TODO: actually implement this
        print("brightess = %f" % brightness)
        if not 0.0 < brightness < 1.0:
            raise ValueError('brightness must be between 0-1, not '
                             + brightness)

        self.brightness = brightness
        self.update()

    @property
    def cmap(self):
        """string: Color map.
        """
        return self.image_visual.cmap

    @cmap.setter
    def cmap(self, cmap):
        self.image_visual.cmap = cmap
        self.update()
