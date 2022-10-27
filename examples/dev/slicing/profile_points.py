import cProfile, pstats, io
from pstats import SortKey

import numpy as np

from napari.layers import Points

np.random.seed(0)

n = 65536
data = np.random.random((n, 2))
s = io.StringIO()

reps = 100

# Profiling
with cProfile.Profile() as pr:
    for k in range(reps):
        layer = Points(data)
        layer._set_view_slice()

sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(0.05)
print(s.getvalue())
# pr.dump_stats("result.pstat")
