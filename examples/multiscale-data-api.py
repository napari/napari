import numpy as np

from napari.layers._multiscale_data import MultiScaleData

data0 = np.random.random((5, 40, 60))
data1 = np.copy(data0[:, ::2, ::2])
data2 = np.copy(data1[:, ::2, ::2])

data = [data0, data1, data2]

ms = MultiScaleData(data)

sliced = ms[4, :, 22:60]

print(sliced.ndim)

indexed = ms[3, np.arange(4), np.arange(4)]

print(indexed.ndim)

ms[4, :, 5:10] = 2

ms.compute_level = -1

print(np.max(ms))

print(np.mean(ms))
