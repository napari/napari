from napari import gui_qt
from napari.layers import Points, LayerGroup
from napari._qt.tree._tree_view import QtLayerTreeView

with gui_qt():
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
    tree = QtLayerTreeView(root)
    model = tree.model()
    tree.show()
