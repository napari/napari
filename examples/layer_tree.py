from napari.qt import get_app, run
from napari.layers import Points
from napari.layers.layergroup import LayerGroup
from napari._qt.tree import QtLayerTreeView
import logging

end = "\033[0m"
Bold = "\033[1m"
Dim = "\033[2m"
ResetDim = "\033[22m"
red = "\033[0;31m"
green = "\033[0;32m"
colorlog_format = f'{green}%(levelname)9s:{end} {Dim}%(name)36s.{ResetDim}{red}%(funcName)-18s{end}{"%(message)s"}'
logging.basicConfig(level=logging.DEBUG, format=colorlog_format)

app = get_app()

tip = Points(name='tip')
lg2 = LayerGroup([Points(name='2')], name="g2")
lg1 = LayerGroup([lg2, Points(name='3'), tip, Points(name='1')], name="g1")
root = LayerGroup(
    [
        lg1,
        Points(name='4'),
        Points(name='5'),
        Points(name='6'),
        Points(name='7'),
        Points(name='8'),
        Points(name='9'),
    ],
    name="root",
)

root.events.reordered.connect(lambda e: print(e.value))

tree = QtLayerTreeView(root)
model = tree.model()
tree.show()

run()
