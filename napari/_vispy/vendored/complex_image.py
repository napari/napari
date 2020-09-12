# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import numpy as np
from .image import ImageVisual
from vispy.gloo import Texture2D
from vispy.visuals.shaders import Function, FunctionChain
from vispy.ext.six import string_types


class ComplexImageVisual(ImageVisual):
    def __init__(
        self,
        data=None,
        method='auto',
        grid=(1, 1),
        cmap='viridis',
        clim='auto',
        gamma=1.0,
        interpolation='nearest',
        complex_mode='phase',
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
        if self._interpolation == 'bilinear':
            texture_interpolation = 'linear'
        else:
            texture_interpolation = 'nearest'

        self._texture = Texture2D(
            np.zeros((1, 1)),
            interpolation=texture_interpolation,
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
        self._complex_mode = value
        self._need_colortransform_update = True
        self.update()

    def _build_color_transform(self, data, clim, gamma, cmap):

        self._complex_mode = 'magnitude'

        _gray_funcs = {
            'magnitude': _complex_mag,
            'phase': _complex_angle,
            'real': _complex_real,
            'imaginary': _complex_imaginary,
        }
        print(cmap.glsl_map)
        if self.complex_mode in _gray_funcs:
            fclim = Function(_apply_clim)
            fclim['clim'] = clim
            fgamma = Function(_apply_gamma)
            fgamma['gamma'] = gamma
            chain = [
                Function(_gray_funcs[self.complex_mode]),
                fclim,
                fgamma,
                Function(cmap.glsl_map),
            ]
            fun = FunctionChain(None, chain)
        elif self.complex_mode == 'colormap':
            fclim = Function(_apply_clim)
            fclim['clim'] = clim
            chain = [
                Function(_gray_funcs['phase']),
                fclim,
                Function(cmap.glsl_map),
            ]
        return fun

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
    float complex_mag(vec2 data) {
        return sqrt(data.x * data.x + data.y * data.y);
    }"""

_complex_angle = """
    float complex_angle(vec2 data) {
        return atan(data.y, data.x);
    }"""

_complex_real = """
    float complex_real(vec2 data) {
        return data.x;
    }"""

_complex_imaginary = """
    float complex_angle(vec2 data) {
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
