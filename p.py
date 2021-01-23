import gc
import os
import weakref

import numpy as np
import psutil

import napari

v = napari.Viewer()


def get_process_memory():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


for i in range(1000):
    d = np.random.rand(1024, 1024)
    r = weakref.ref(v.add_image(d))
    dr = weakref.ref(d)
    del d
    v.layers.pop(0)
    gc.collect()
    print("r is ", r(), "d is ", type(dr()))
    print(f"Mem used after {i:3} layers: {get_process_memory():0.2f} (MB)")
