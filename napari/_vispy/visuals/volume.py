from vispy.scene.visuals import Volume as BaseVolume

from napari._vispy.visuals.util import TextureMixin
from napari.layers.labels._labels_constants import IsoCategoricalGradientMode

FUNCTION_DEFINITIONS = """
uniform bool u_iso_gradient;

// the tolerance for testing equality of floats with floatEqual and floatNotEqual
const float equality_tolerance = 1e-8;

bool floatNotEqual(float val1, float val2)
{
    // check if val1 and val2 are not equal
    bool not_equal = abs(val1 - val2) > equality_tolerance;

    return not_equal;
}

bool floatEqual(float val1, float val2)
{
    // check if val1 and val2 are equal
    bool equal = abs(val1 - val2) < equality_tolerance;

    return equal;
}


// the background value for the iso_categorical shader
const float categorical_bg_value = 0;

int detectAdjacentBackground(float val_neg, float val_pos)
{
    // determine if the adjacent voxels along an axis are both background
    int adjacent_bg = int( floatEqual(val_neg, categorical_bg_value) );
    adjacent_bg = adjacent_bg * int( floatEqual(val_pos, categorical_bg_value) );
    return adjacent_bg;
}

vec3 calculateGradient(vec3 loc, vec3 step, out int n_bg_borders) {
    // calculate gradient within the volume by finite differences

    n_bg_borders = 0;
    vec3 G = vec3(0.0);

    float prev;
    float next;

    prev = colorToVal($get_data(loc - vec3(step.x, 0, 0)));
    next = colorToVal($get_data(loc + vec3(step.x, 0, 0)));
    n_bg_borders += detectAdjacentBackground(prev, next);
    G.x = next - prev;

    prev = colorToVal($get_data(loc - vec3(0, step.y, 0)));
    next = colorToVal($get_data(loc + vec3(0, step.y, 0)));
    n_bg_borders += detectAdjacentBackground(prev, next);
    G.y = next - prev;

    prev = colorToVal($get_data(loc - vec3(0, 0, step.z)));
    next = colorToVal($get_data(loc + vec3(0, 0, step.z)));
    n_bg_borders += detectAdjacentBackground(prev, next);
    G.z = next - prev;

    return G;
}

vec3 calculateIsotropicGradient(vec3 loc, vec3 step) {
    // calculate gradient within the volume by finite differences
    // using a 3D sobel-feldman convolution kernel

    // the kernel here is a 3x3 cube, centered on the sample at `loc`
    // the kernel for G.z looks like this:

    // [ +1 +2 +1 ]
    // [ +2 +4 +2 ]    <-- "loc - step.z" is in the center
    // [ +1 +2 +1 ]

    // [  0  0  0 ]
    // [  0  0  0 ]    <-- "loc" is in the center
    // [  0  0  0 ]

    // [ -1 -2 -1 ]
    // [ -2 -4 -2 ]    <-- "loc + step.z" is in the center
    // [ -1 -2 -1 ]

    // kernels for G.x and G.y similar, but transposed
    // see https://en.wikipedia.org/wiki/Sobel_operator#Extension_to_other_dimensions

    vec3 G = vec3(0.0);

    for (int i=-1; i <= 1; i++) {
        for (int j=-1; j <= 1; j++) {
            for (int k=-1; k <= 1; k++) {
                float val = colorToVal($get_data(loc + vec3(i, j, k) * step));
                G.x += val * -float(i) *
                    (1 + float(j == 0 || k == 0) + 2 * float(j == 0 && k == 0));
                G.y += val * -float(j) *
                    (1 + float(i == 0 || k == 0) + 2 * float(i == 0 && k == 0));
                G.z += val * -float(k) *
                    (1 + float(i == 0 || j == 0) + 2 * float(i == 0 && j == 0));
            }
        }
    }

    return G;
}

vec4 calculateShadedCategoricalColor(vec4 betterColor, vec3 loc, vec3 step)
{
    // Calculate color by incorporating ambient and diffuse lighting
    vec4 color0 = $get_data(loc);
    vec4 color1;
    vec4 color2;
    float val0 = colorToVal(color0);
    float val1 = 0;
    float val2 = 0;
    int n_bg_borders = 0;

    // View direction
    vec3 V = normalize(view_ray);

    // Calculate normal vector from gradient
    vec3 N;
    if (u_iso_gradient) {
        N = calculateIsotropicGradient(loc, step);
    } else {
        N = calculateGradient(loc, step, n_bg_borders);
    }

    // Normalize and flip normal so it points towards viewer
    N = normalize(N);
    float Nselect = float(dot(N,V) > 0.0);
    N = (2.0*Nselect - 1.0) * N;  // ==  Nselect * N - (1.0-Nselect)*N;

    // Init colors
    vec4 ambient_color = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 diffuse_color = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 final_color;

    // todo: allow multiple light, define lights on viewvox or subscene
    int nlights = 1;
    for (int i=0; i<nlights; i++)
    {
        // Get light direction (make sure to prevent zero devision)
        vec3 L = normalize(view_ray);  //lightDirs[i];
        float lightEnabled = float( length(L) > 0.0 );
        L = normalize(L+(1.0-lightEnabled));

        // Calculate lighting properties
        float lambertTerm = clamp( dot(N,L), 0.0, 1.0 );
        if (n_bg_borders > 0) {
            // to fix dim pixels due to poor normal estimation,
            // we give a default lambda to pixels surrounded by background
            lambertTerm = 0.5;
        }

        // Calculate mask
        float mask1 = lightEnabled;

        // Calculate colors
        ambient_color +=  mask1 * u_ambient;  // * gl_LightSource[i].ambient;
        diffuse_color +=  mask1 * lambertTerm;
    }

    // Calculate final color by componing different components
    final_color = betterColor * ( ambient_color + diffuse_color);
    final_color.a = betterColor.a;

    // Done
    return final_color;
}
"""

ISO_CATEGORICAL_SNIPPETS = {
    'before_loop': """
        vec4 color3 = vec4(0.0);  // final color
        vec3 dstep = 1.5 / u_shape;  // step to sample derivative, set to match iso shader
        gl_FragColor = vec4(0.0);
        bool discard_fragment = true;
        vec4 label_id = vec4(0.0);
        """,
    'in_loop': """
        // check if value is different from the background value
        if ( floatNotEqual(val, categorical_bg_value) ) {
            // Take the last interval in smaller steps
            vec3 iloc = loc - step;
            for (int i=0; i<10; i++) {
                label_id = $get_data(iloc);
                color = sample_label_color(label_id.r);
                if (floatNotEqual(color.a, 0) ) {
                    // fully transparent color is considered as background, see napari/napari#5227
                    // when the value mapped to non-transparent color is reached
                    // calculate the shaded color (apply lighting effects)
                    color = calculateShadedCategoricalColor(color, iloc, dstep);
                    gl_FragColor = color;

                    // set the variables for the depth buffer
                    frag_depth_point = iloc * u_shape;
                    discard_fragment = false;

                    iter = nsteps;
                    break;
                }
                iloc += step * 0.1;
            }
        }
        """,
    'after_loop': """
        if (discard_fragment)
            discard;
        """,
}

TRANSLUCENT_CATEGORICAL_SNIPPETS = {
    'before_loop': """
        vec4 color3 = vec4(0.0);  // final color
        gl_FragColor = vec4(0.0);
        bool discard_fragment = true;
        vec4 label_id = vec4(0.0);
        """,
    'in_loop': """
        // check if value is different from the background value
        if ( floatNotEqual(val, categorical_bg_value) ) {
            // Take the last interval in smaller steps
            vec3 iloc = loc - step;
            for (int i=0; i<10; i++) {
                label_id = $get_data(iloc);
                color = sample_label_color(label_id.r);
                if (floatNotEqual(color.a, 0) ) {
                    // fully transparent color is considered as background, see napari/napari#5227
                    // when the value mapped to non-transparent color is reached
                    // calculate the color (apply lighting effects)
                    gl_FragColor = color;

                    // set the variables for the depth buffer
                    frag_depth_point = iloc * u_shape;
                    discard_fragment = false;

                    iter = nsteps;
                    break;
                }
                iloc += step * 0.1;
            }
        }
        """,
    'after_loop': """
        if (discard_fragment)
            discard;
        """,
}

shaders = BaseVolume._shaders.copy()
before, after = shaders['fragment'].split('void main()')
shaders['fragment'] = before + FUNCTION_DEFINITIONS + 'void main()' + after

rendering_methods = BaseVolume._rendering_methods.copy()
rendering_methods['iso_categorical'] = ISO_CATEGORICAL_SNIPPETS
rendering_methods['translucent_categorical'] = TRANSLUCENT_CATEGORICAL_SNIPPETS


class Volume(TextureMixin, BaseVolume):
    """This class extends the vispy Volume visual to add categorical isosurface rendering."""

    # add the new rendering method to the snippets dict
    _shaders = shaders
    _rendering_methods = rendering_methods

    def __init__(self, *args, **kwargs) -> None:  # type: ignore [no-untyped-def]
        super().__init__(*args, **kwargs)
        self.unfreeze()
        self.iso_gradient_mode = IsoCategoricalGradientMode.FAST.value
        self.freeze()

    @property
    def iso_gradient_mode(self) -> str:
        return str(self._iso_gradient_mode)

    @iso_gradient_mode.setter
    def iso_gradient_mode(self, value: str) -> None:
        self._iso_gradient_mode = IsoCategoricalGradientMode(value)
        self.shared_program['u_iso_gradient'] = (
            self._iso_gradient_mode == IsoCategoricalGradientMode.SMOOTH
        )
        self.update()
