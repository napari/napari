import gc
import sys
import weakref

import numpy as np

import napari

viewer = napari.viewer.Viewer()
lr = weakref.ref(viewer.add_image(np.random.rand(512, 512)))
dr = weakref.ref(lr().data)

viewer.layers.pop()
print(gc.collect(), "unreachable objects after collection")
print(gc.collect(), "unreachable objects after collection")
print("l is ", lr(), "d is ", dr())

if lr():
    print(f"\n{sys.getrefcount(lr()) - 1} remaining references to {lr()}")
if dr() is not None:
    print(
        f"\n{sys.getrefcount(dr()) - 1} remaining references to {type(dr())}"
    )
    print(gc.get_referrers(dr()))
