from superqt import QDoubleSlider
from superqt.qtcompat import QT_VERSION
from superqt.qtcompat.QtWidgets import QSlider  # noqa

# here until we can debug why labeled sliders render differently on 5.12
if tuple(int(x) for x in QT_VERSION.split(".")) >= (5, 14):
    from superqt import QLabeledDoubleSlider as QDoubleSlider  # noqa
    from superqt import QLabeledSlider as QSlider  # noqa
