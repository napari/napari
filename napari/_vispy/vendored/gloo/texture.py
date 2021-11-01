# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------


import numpy as np
import warnings

F64_PRECISION_WARNING = ("GPUs can't support floating point data with more "
                         "than 32-bits, precision will be lost due to "
                         "downcasting to 32-bit float.")


def should_cast_to_f32(data_dtype):
    """Check if data type is floating point with more than 32-bits."""
    data_dtype = np.dtype(data_dtype)
    is_floating = np.issubdtype(data_dtype, np.floating)
    gt_float32 = data_dtype.itemsize > 4
    if is_floating and gt_float32:
        # OpenGL can't support floating point numbers greater than 32-bits
        warnings.warn(F64_PRECISION_WARNING)
        return True
    return False
