from vispy.visuals.filters import Filter


class ClampSizeFilter(Filter):
    """
    Clamp the size of the points in canvas pixels.
    """

    VERT_SHADER = """
        void clamp_size() {
            if ($active == 1) {
                gl_PointSize = clamp(gl_PointSize, $min, $max);
            }
        }
    """

    def __init__(self, min_size=0, max_size=10000, active=True):
        super().__init__(vcode=self.VERT_SHADER, vpos=10000)
        self.min_size = min_size
        self.max_size = max_size
        self.active = active

    @property
    def min_size(self):
        return self._min_size

    @min_size.setter
    def min_size(self, value):
        self._min_size = float(value)
        self.vshader['min'] = self._min_size

    @property
    def max_size(self):
        return self._max_size

    @max_size.setter
    def max_size(self, value):
        self._max_size = float(value)
        self.vshader['max'] = self._max_size

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        self._active = bool(value)
        self.vshader['active'] = self._active
