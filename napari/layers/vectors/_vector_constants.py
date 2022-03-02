from enum import auto

from ...utils.misc import StringEnum


class VectorsProjection(StringEnum):
    """
    veconimensional slices into a single
      point slice.
          * SLICE: ignore slice thickness, only using the dims point
          * ADDITIVE: include all vectors whose starting position is within the thick slice
    """

    SLICE = auto()
    ADDITIVE = auto()
