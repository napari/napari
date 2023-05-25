import gc
import os
import weakref

import numpy as np
import objgraph
import psutil
import qtpy

import napari

process = psutil.Process(os.getpid())
viewer = napari.Viewer()

print("mem", process.memory_info().rss)

for _ in range(0):
    print(viewer.add_image(np.random.random((60, 1000, 1000))).name)
for _ in range(2):
    print(viewer.add_labels((np.random.random((2, 1000, 1000)) * 10).astype(np.uint8)).name)

print("mem", process.memory_info().rss)

# napari.run()

print("controls", viewer.window.qt_viewer.controls.widgets)
li = weakref.ref(viewer.layers[0])
data_li = weakref.ref(li()._data)
controls = weakref.ref(viewer.window.qt_viewer.controls.widgets[li()])
objgraph.show_backrefs(li(), filename="base.png")
del viewer.layers[0]
qtpy.QtGui.QGuiApplication.processEvents()
gc.collect()
gc.collect()
print(li())
objgraph.show_backrefs(li(), max_depth=10, filename="test.png", refcounts=True)
objgraph.show_backrefs(controls(), max_depth=10, filename="controls.png", refcounts=True)
objgraph.show_backrefs(data_li(), max_depth=10,  filename="test_data.png")
print("controls", viewer.window.qt_viewer.controls.widgets)
print("controls", gc.get_referrers(controls()))
print("controls", controls().parent())
#print("controls", controls().parent().indexOf(controls()))
print(gc.get_referrers(li()))
print(gc.get_referrers(li())[1])
print(gc.get_referrers(gc.get_referrers(gc.get_referrers(li())[0])))
res = gc.get_referrers(gc.get_referrers(gc.get_referrers(li())[0])[0])
print(res)
#print(type(res[0]))
