import os

# Should not be imported unless async is defined.
assert os.getenv("NAPARI_ASYNC", "0") != "0"
