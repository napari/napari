# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import numpy as np
from .image import ImageVisual
from vispy.gloo import Texture2D
from vispy.visuals.shaders import Function, FunctionChain
from vispy.ext.six import string_types


class ComplexImageVisual(ImageVisual):
    _complex_modes = {'magnitude', 'phase', 'real', 'imaginary'}

    def __init__(
        self,
        data=None,
        method='auto',
        grid=(1, 1),
        cmap='viridis',
        clim='auto',
        gamma=1.0,
        interpolation='nearest',
        complex_mode='magnitude',
        **kwargs,
    ):
        self._x = False
        self._complex_mode = complex_mode
        self._texture_i = Texture2D(
            np.zeros((1, 1)),
            interpolation=interpolation,
            internalformat='r32f',
            format='luminance',
        )
        super().__init__()

        interp = 'linear' if self._interpolation == 'bilinear' else 'nearest'
        self._texture = Texture2D(
            np.zeros((1, 1)),
            interpolation=interp,
            internalformat='r32f',
            format='luminance',
        )
        self._interpolation_names = ['nearest', 'bilinear']

    def _build_interpolation(self):
        super()._build_interpolation()
        self._data_lookup_fn['texture_i'] = self._texture_i

    def _build_texture(self):
        data = self._data
        if data.dtype == np.complex128:
            data = data.astype(np.complex64)

        clim = self._clim
        if isinstance(clim, string_types) and clim == 'auto':
            clim = np.min(data), np.max(data)
        self._clim = np.asarray(clim, dtype=np.float32)

        self._texture.set_data(data.real)
        self._texture_i.set_data(data.imag)
        self._need_texture_upload = False
        self._texture_limits = (-np.inf, np.inf)  # hack?

    @property
    def clim_normalized(self):
        return self.clim

    @property
    def complex_mode(self):
        return self._complex_mode

    @complex_mode.setter
    def complex_mode(self, value):
        if value not in self._complex_modes:
            raise ValueError(
                "complex_mode must be one of %s"
                % ', '.join(self._complex_modes)
            )
        if self._complex_mode != value:
            self._complex_mode = value
            self._need_colortransform_update = True
            self.update()

    def _build_color_transform(self, data, clim, gamma, cmap):

        mode_funcs = {
            'magnitude': _complex_mag,
            'phase': _complex_angle,
            'real': _complex_real,
            'imaginary': _complex_imaginary,
        }

        fclim = Function(_apply_clim)
        fgamma = Function(_apply_gamma)
        fclim['clim'] = clim
        fgamma['gamma'] = gamma
        chain = [
            Function(mode_funcs[self.complex_mode]),
            fclim,
            fgamma,
            Function(cmap.glsl_map),
        ]
        return FunctionChain(None, chain)

    _texture_lookup = """
        vec2 texture_lookup(vec2 texcoord) {
            if(texcoord.x < 0.0 || texcoord.x > 1.0 ||
            texcoord.y < 0.0 || texcoord.y > 1.0) {
                discard;
            }
            vec4 real = texture2D($texture, texcoord);
            vec4 imag = texture2D($texture_i, texcoord);
            return vec2(real.x, imag.x);
        }"""


_complex_mag = """
    float comp2float(vec2 data) {
        return sqrt(data.x * data.x + data.y * data.y);
    }"""

_complex_angle = """
    float comp2float(vec2 data) {
        return atan(data.y, data.x);
    }"""

_complex_real = """
    float comp2float(vec2 data) {
        return data.x;
    }"""

_complex_imaginary = """
    float comp2float(vec2 data) {
        return data.y;
    }"""

_apply_clim = """
    float apply_clim(float data) {
        data = data - $clim.x;
        data = data / ($clim.y - $clim.x);
        return max(data, 0);
    }"""

_apply_gamma = """
    float apply_gamma(float data) {
        return pow(data, $gamma);
    }"""
