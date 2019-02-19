import os


_matplotlib_list_file = os.path.join(os.path.dirname(__file__),
                                     'matplotlib_cmaps.txt')
with open(_matplotlib_list_file) as fin:
    matplotlib_colormaps = [line.rstrip() for line in fin]
