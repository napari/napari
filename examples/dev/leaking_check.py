import os
import psutil
import weakref
import gc
import objgraph

import napari
import numpy as np

process = psutil.Process(os.getpid())
viewer = napari.Viewer()

print("mem", process.memory_info().rss)

for i in range(15):
    print(viewer.add_image(np.random.random((60, 1000, 1000))).name)

print("mem", process.memory_info().rss)

napari.run()

li = weakref.ref(viewer.layers[0])
del viewer.layers[0]

objgraph.show_backrefs(li(), filename="test.png")
ref_li = gc.get_referrers(li())
ref_li2 = gc.get_referrers(ref_li[0])
