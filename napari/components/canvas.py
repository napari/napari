from typing import Any, Dict, OrderedDict, Tuple

from pydantic import Field

from napari.components import LayerList
from napari.layers import Layer
from napari.utils.events import EventedModel


class Canvas(EventedModel):
    layers: LayerList = Field(default_factory=LayerList, allow_mutation=False)
    slice_box: OrderedDict[str, Tuple[Tuple[float, float], ...]]
    layer_to_slice: Dict[Layer, Any]

    def update_slices(self) -> None:
        for layer in self.layers:
            self.layer_to_slice[layer] = layer.get_slice(self.slice_box)

    @property
    def ndisplay(self) -> int:
        return len(self.slice_box)

    @property
    def extent(
        self,
    ) -> OrderedDict[str, Tuple[Tuple[float, float, float], ...]]:
        pass

    @property
    def ndim(self) -> int:
        return len(self.layers.axis_labels)

    @property
    def displayed(self) -> Tuple[str, ...]:
        return tuple(self.slice_box)

    @property
    def not_displayed(self) -> Tuple[str, ...]:
        return tuple(self.layers.axis_labels - set(self.displayed))

    @property
    def nsteps(self) -> OrderedDict[str, Tuple[float, ...]]:
        pass
