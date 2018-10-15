from abc import ABC, abstractmethod
import weakref

from ._visual_wrapper import VisualWrapper


class Layer(VisualWrapper, ABC):
    def __init__(self, central_node):
        super().__init__(central_node)
        self._selected = False
        self._viewer = None
        self._on_select_hook = []

    @property
    @abstractmethod
    def data(self):
        # user writes own docstring
        raise NotImplementedError()

    @data.setter
    @abstractmethod
    def data(self, data):
        raise NotImplementedError()

    @abstractmethod
    def _get_shape(self):
        raise NotImplementedError()

    @abstractmethod
    def _refresh(self):
        raise NotImplementedError()

    @property
    def ndim(self):
        """int: Number of dimensions in the data.
        """
        return len(self.shape)

    @property
    def shape(self):
        """tuple of int: Shape of the data.
        """
        return self._get_shape()

    @property
    def selected(self):
        """boolean: Wether this layer is selected or not.
        """
        return self._selected

    @selected.setter
    def selected(self, selected):
        if selected == self.selected:
            return
        self._selected = selected
        for callback in self._on_select_hook:
            callback(selected)

    @property
    def _qt(self):
        """PyQt5.QWidget or None: Widget, if any, inserted when
        solely this layer is selected.
        """
        return None

    @property
    def viewer(self):
        """Viewer: Parent viewer widget.
        """
        if self._viewer is not None:
            return self._viewer()

    @viewer.setter
    def viewer(self, viewer):
        if viewer is None:
            self._viewer = None
            parent = None
        else:
            self._viewer = weakref.ref(viewer)
            parent = viewer._qt.view.scene


        vt = self._node.transforms.visual_transform
        trs = vt.transforms
        self._set_parent(parent)
        vt.transforms = trs
        self._after_set_viewer(viewer)

    def _after_set_viewer(self, viewer):
        """Triggered after a new viewer is set.

        Parameters
        ----------
        viewer : Viewer
            Parent viewer.
        """
        self.refresh()

    def _set_parent(self, parent):
        """Set the parent node.

        Parameters
        ----------
        parent : vispy.scene.Node
            Parent node.
        """
        self._node.parent = parent

    def _set_view_slice(self, indices):
        """Called whenever the sliders change. Sets the
        current view given a specific slice to view.

        Parameters
        ----------
        indices : sequence of int or slice
            Indices that make up the slice.
        """

    def refresh(self):
        """Fully refreshes the layer.
        """
        self._refresh()
