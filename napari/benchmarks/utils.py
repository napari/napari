from functools import lru_cache

import numpy as np
from skimage import morphology


class Skiper:
    def __init__(self, func):
        self.func = func

    def __contains__(self, item):
        return self.func(item)


@lru_cache
def gen_blobs(shape, dtype, blob_count=144):
    np.random.seed(1)
    balls_ = np.zeros(shape, dtype=dtype)
    gen = np.random.default_rng(1)
    points = (
        gen.random((len(shape), blob_count)) * np.array(shape).reshape((-1, 1))
    ).astype(int)
    values = gen.integers(
        np.iinfo(dtype).min, np.iinfo(dtype).max, size=blob_count, dtype=dtype
    )
    sigma = int(np.array(shape).max() / (4.0 * blob_count ** (1 / len(shape))))
    if len(shape) == 2:
        ball = morphology.disk(sigma)
    else:
        ball = morphology.ball(sigma)
    for j, point in enumerate(points.T):
        slice_im = []
        slice_ball = []
        for i, p in enumerate(point):
            slice_im.append(
                slice(max(0, p - sigma), min(shape[i], p + sigma + 1))
            )
            ball_base = max(0, sigma - p)
            bal_end = slice_im[-1].stop - slice_im[-1].start + ball_base
            slice_ball.append(slice(ball_base, bal_end))

        balls_[tuple(slice_im)][ball[tuple(slice_ball)] > 0] = values[j]

    return balls_
