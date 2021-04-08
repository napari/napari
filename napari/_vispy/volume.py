import numpy as np
from vispy.color import get_colormap
from vispy.scene.visuals import create_visual_node
from vispy.visuals.shaders import Function

from .vendored import VolumeVisual as BaseVolumeVisual
from .vendored.volume import FRAG_SHADER, frag_dict

BaseVolume = create_visual_node(BaseVolumeVisual)

ATTENUATED_MIP_SNIPPETS = dict(
    before_loop="""
        float maxval = -99999.0; // The maximum encountered value
        float sumval = 0.0; // The sum of the encountered values
        float scaled = 0.0; // The scaled value
        int maxi = 0;  // Where the maximum value was encountered
        vec3 maxloc = vec3(0.0);  // Location where the maximum value was encountered
        """,
    in_loop="""
        sumval = sumval + val;
        scaled = val * exp(-u_attenuation * (sumval - 1) / u_relative_step_size);
        if( scaled > maxval ) {
            maxval = scaled;
            maxi = iter;
            maxloc = loc;
        }
        """,
    after_loop="""
        gl_FragColor = applyColormap(maxval);
        """,
)
ATTENUATED_MIP_FRAG_SHADER = FRAG_SHADER.format(**ATTENUATED_MIP_SNIPPETS)

frag_dict['attenuated_mip'] = ATTENUATED_MIP_FRAG_SHADER

MINIP_SNIPPETS = dict(
    before_loop="""
        float minval = 99999.0; // The minimum encountered value
        int mini = 0;  // Where the minimum value was encountered
        """,
    in_loop="""
        if( val < minval ) {
            minval = val;
            mini = iter;
        }
        """,
    after_loop="""
        // Refine search for min value
        loc = start_loc + step * (float(mini) - 0.5);
        for (int i=0; i<10; i++) {
            minval = min(minval, $sample(u_volumetex, loc).g);
            loc += step * 0.1;
        }
        gl_FragColor = applyColormap(minval);
        """,
)

MINIP_FRAG_SHADER = FRAG_SHADER.format(**MINIP_SNIPPETS)

frag_dict['minip'] = MINIP_FRAG_SHADER

AVG_SNIPPETS = dict(
    before_loop="""
        float n = 0; // Counter for encountered values
        float meanval = 0.0; // The mean of encountered values
        float prev_mean = 0.0; // Variable to store the previous incremental mean
        """,
    in_loop="""
        // Incremental mean value used for numerical stability
        n += 1; // Increment the counter
        prev_mean = meanval; // Update the mean for previous iteration
        meanval = prev_mean + (val - prev_mean) / n; // Calculate the mean
        """,
    after_loop="""
        // Apply colormap on mean value
        gl_FragColor = applyColormap(meanval);
        """,
)
AVG_FRAG_SHADER = FRAG_SHADER.format(**AVG_SNIPPETS)
frag_dict['average'] = AVG_FRAG_SHADER


# Custom volume class is needed for better 3D rendering
class Volume(BaseVolume):
    def __init__(self, *args, **kwargs):
        self._attenuation = 1.0
        self._bounding_box_lims = None
        super().__init__(*args, **kwargs)

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        self._cmap = get_colormap(cmap)
        self.shared_program.frag['cmap'] = Function(self._cmap.glsl_map)
        # Colormap change fix
        self.shared_program['texture2D_LUT'] = (
            self.cmap.texture_lut()
            if (hasattr(self.cmap, 'texture_lut'))
            else None
        )
        self.update()

    @property
    def threshold(self):
        """The threshold value to apply for the isosurface render method."""
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        # Fix for #1399, should be fixed in the VisPy threshold setter
        self._threshold = float(value)
        self.shared_program['u_threshold'] = self._threshold
        self.update()

    @property
    def attenuation(self):
        """The attenuation value to apply for the attenuated mip render method."""
        return self._attenuation

    @attenuation.setter
    def attenuation(self, value):
        # Fix for #1399, should be fixed in the VisPy threshold setter
        self._attenuation = float(value)
        self.shared_program['u_attenuation'] = self._attenuation
        self.update()

    @property
    def _bounding_box(self):
        return self._bounding_box_lims

    @_bounding_box.setter
    def _bounding_box(self, bounding_box):
        if bounding_box is not None:
            self._bounding_box_lims = np.asarray(
                bounding_box, dtype=np.float32
            )
            self.shared_program['u_bbox_x_min'] = self._bounding_box_lims[0, 0]
            self.shared_program['u_bbox_x_max'] = self._bounding_box_lims[0, 1]

            self.shared_program['u_bbox_y_min'] = self._bounding_box_lims[1, 0]
            self.shared_program['u_bbox_y_max'] = self._bounding_box_lims[1, 1]

            self.shared_program['u_bbox_z_min'] = self._bounding_box_lims[2, 0]
            self.shared_program['u_bbox_z_max'] = self._bounding_box_lims[2, 1]

            self.update()
        else:
            self._bounding_box_lims = bounding_box

    def _initialize_bounding_box(self):
        """Initialize a bounding box to the full extent of the volume texture"""
        # save to private variable to bypass self.update() call
        bounding_box_xlim = 0, self._vol_shape[2]
        bounding_box_ylim = 0, self._vol_shape[1]
        bounding_box_zlim = 0, self._vol_shape[0]

        self._bounding_box_lims = np.asarray(
            (
                bounding_box_xlim,
                bounding_box_ylim,
                bounding_box_zlim,
            )
        )

    def set_data(self, vol, clim=None, bounding_box=None, copy=True):
        """Set the volume data.

        Parameters
        ----------
        vol : ndarray
            The 3D volume.
        clim : tuple | None
            Colormap limits to use. None will use the min and max values.
        copy : bool | True
            Whether to copy the input volume prior to applying clim normalization.
        """
        # Check volume
        if not isinstance(vol, np.ndarray):
            raise ValueError('Volume visual needs a numpy array.')
        if not ((vol.ndim == 3) or (vol.ndim == 4 and vol.shape[-1] <= 4)):
            raise ValueError('Volume visual needs a 3D image.')

        # Handle clim
        if clim is not None:
            clim = np.array(clim, float)
            if not (clim.ndim == 1 and clim.size == 2):
                raise ValueError('clim must be a 2-element array-like')
            self._clim = tuple(clim)
        if self._clim is None:
            self._clim = vol.min(), vol.max()

        # store clims used to normalize _tex data for use in clim_normalized
        self._texture_limits = self._clim
        # store volume in case it needs to be renormalized by clim.setter
        self._last_data = vol
        self.shared_program['clim'] = self.clim_normalized

        # Apply clim (copy data by default... see issue #1727)
        vol = np.array(vol, dtype='float32', copy=copy)
        if self._clim[1] == self._clim[0]:
            if self._clim[0] != 0.0:
                vol *= 1.0 / self._clim[0]
        elif self._clim[0] > self._clim[1]:
            vol *= -1
            vol += self._clim[1]
            vol /= self._clim[1] - self._clim[0]
        else:
            vol -= self._clim[0]
            vol /= self._clim[1] - self._clim[0]

        # Apply to texture
        self._tex.set_data(vol)  # will be efficient if vol is same shape
        self.shared_program['u_shape'] = (
            vol.shape[2],
            vol.shape[1],
            vol.shape[0],
        )

        shape = vol.shape[:3]
        if self._vol_shape != shape:
            self._vol_shape = shape
            self._need_vertex_update = True
        self._vol_shape = shape

        if self._bounding_box is None and bounding_box is None:
            # vispy node is initialized before data model so we need to provide
            # an initial bounding box. this gets updated when the Image layer is
            # initialized
            self._initialize_bounding_box()
        elif bounding_box is not None:
            # we re-order the axes to match vispy
            self._bounding_box = bounding_box[[2, 1, 0], :]

        self.shared_program['u_bbox_x_min'] = self._bounding_box[0, 0]
        self.shared_program['u_bbox_x_max'] = self._bounding_box[0, 1]
        self.shared_program['u_bbox_y_min'] = self._bounding_box[1, 0]
        self.shared_program['u_bbox_y_max'] = self._bounding_box[1, 1]
        self.shared_program['u_bbox_z_min'] = self._bounding_box[2, 0]
        self.shared_program['u_bbox_z_max'] = self._bounding_box[2, 1]

        # Get some stats
        self._kb_for_texture = np.prod(self._vol_shape) / 1024
