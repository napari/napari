import napari
from napari._qt.qt_layertree import QtLayerTree
from napari.layers import LayerGroup, Points, Shapes

with napari.gui_qt():
    lg2 = LayerGroup([Points(), Points()])
    lg_root = LayerGroup([Points(), lg2, Shapes()])
    tree = QtLayerTree(lg_root)
    tree.show()
