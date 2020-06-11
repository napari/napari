# -------------------------------------------------------------------------------
# Name:     Arboretum
# Purpose:  Dockable widget, and custom track visualization layers for Napari,
#           to cell/object track data.
#
# Authors:  Alan R. Lowe (arl) a.lowe@ucl.ac.uk
#
# License:  See LICENSE.md
#
# Created:  01/05/2020
# -------------------------------------------------------------------------------

__author__ = 'Alan R. Lowe'
__email__ = 'code@arlowe.co.uk'

from vispy.visuals.filters.base_filter import Filter
from vispy.gloo import VertexBuffer

from typing import Union, List

import numpy as np


class TrackShader(Filter):
    """ TrackShader

    Custom vertex and fragment shaders for visualizing tracks quickly with
    vispy. The central assumption is that the tracks are rendered as a
    continuous vispy Line segment, with connections and colors defined when
    the visual is created.

    The shader simply changes the visibility and/or fading of the data according
    to the current_time and the associate time metadata for each vertex. This
    is scaled according to the tail length. Points ahead of the current time
    are rendered with alpha set to zero.

    Can also apply a mask directly to the shader to slice data

    Parameters
    ----------

        current_time: int, float
            the current time, which is typically the frame index, although this
            can be an arbitrary float
        tail_length: int, float
            the upper limit on length of the 'tail'
        use_fade: bool
            this will enable/disable tail fading with time
        vertex_time: 1D array, list
            a vector describing the time associated with each vertex
        vertex_mask: 1D array, list
            a vector describing whether to mask each vertex

    TODO
    ----
        - the track is still displayed, albeit with fading, once the track has
         finished but is still within the 'tail_length' window. Should it
         disappear?
        - check the shader positioning within the GL pipeline

    """

    VERT_SHADER = """
        varying vec4 v_track_color;
        void apply_track_shading() {

            float alpha;

            if ($a_vertex_time > $current_time) {
                // this is a hack to minimize the frag shader rendering ahead
                // of the current time point due to interpolation
                if ($a_vertex_time <= $current_time + 1){
                    alpha = -100.;
                } else {
                    alpha = 0.;
                }
            } else {
                // fade the track into the temporal distance, scaled by the
                // maximum tail length from the gui
                float fade = ($current_time - $a_vertex_time) / $tail_length;
                alpha = clamp(1.0-fade, 0.0, 1.0);
            }

            // when use_fade is disabled, the entire track is visible
            if ($use_fade == 0) {
                alpha = 1.0;
            }

            // finally, if we're applying a mask (for e.g. slicing ND data),
            // do it here. THIS WILL OVERIDE the time based vertex shading and
            // set the vertex alpha to zero

            alpha = (1.-$a_vertex_mask) * alpha;

            // set the vertex alpha according to the fade
            v_track_color.a = alpha;
        }
    """

    FRAG_SHADER = """
        varying vec4 v_track_color;
        void apply_track_shading() {
            // interpolate
            gl_FragColor.a = v_track_color.a;
        }
    """

    def __init__(
        self,
        current_time=0,
        tail_length=30,
        use_fade: bool = True,
        vertex_time: Union[List, np.ndarray] = [],
        vertex_mask: Union[List, np.ndarray] = [],
    ):

        super().__init__(
            vcode=self.VERT_SHADER, vpos=3, fcode=self.FRAG_SHADER, fpos=9
        )

        self.current_time = current_time
        self.tail_length = tail_length
        self.use_fade = use_fade
        self.vertex_time = vertex_time
        self.vertex_mask = vertex_mask

    @property
    def current_time(self) -> Union[int, float]:
        return self._current_time

    @current_time.setter
    def current_time(self, n: Union[int, float]):
        self._current_time = n
        if isinstance(n, slice):
            n = np.max(self._vertex_time)
        self.vshader['current_time'] = float(n)

    @property
    def use_fade(self) -> bool:
        return self._use_fade

    @use_fade.setter
    def use_fade(self, value: bool):
        self._use_fade = value
        self.vshader['use_fade'] = float(value)

    @property
    def tail_length(self) -> Union[int, float]:
        return self._tail_length

    @tail_length.setter
    def tail_length(self, l: Union[int, float]):
        self._tail_length = l
        self.vshader['tail_length'] = float(l)

    def _attach(self, visual):
        super(TrackShader, self)._attach(visual)
        self.vshader['a_vertex_time'] = VertexBuffer(self.vertex_time)
        self.vshader['a_vertex_mask'] = VertexBuffer(self.vertex_mask)

    @property
    def vertex_time(self):
        return self._vertex_time

    @vertex_time.setter
    def vertex_time(self, v_time):
        self._vertex_time = np.array(v_time).reshape(-1, 1).astype(np.float32)

    @property
    def vertex_mask(self):
        return self._vertex_mask

    @vertex_mask.setter
    def vertex_mask(self, v_mask):
        if not v_mask:
            v_mask = np.zeros(self.vertex_time.shape, dtype=np.float32)
        self._vertex_mask = v_mask
