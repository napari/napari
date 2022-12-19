from vispy.scene.visuals import Volume as BaseVolume

FUNCTION_DEFINITIONS = """
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

vec4 calculateCategoricalColor(vec4 betterColor, vec3 loc, vec3 step)
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

    // calculate normal vector from gradient
    vec3 N; // normal
    color1 = $get_data(loc+vec3(-step[0],0.0,0.0));
    color2 = $get_data(loc+vec3(step[0],0.0,0.0));
    val1 = colorToVal(color1);
    val2 = colorToVal(color2);
    N[0] = val1 - val2;
    n_bg_borders += detectAdjacentBackground(val1, val2);

    color1 = $get_data(loc+vec3(0.0,-step[1],0.0));
    color2 = $get_data(loc+vec3(0.0,step[1],0.0));
    val1 = colorToVal(color1);
    val2 = colorToVal(color2);
    N[1] = val1 - val2;
    n_bg_borders += detectAdjacentBackground(val1, val2);

    color1 = $get_data(loc+vec3(0.0,0.0,-step[2]));
    color2 = $get_data(loc+vec3(0.0,0.0,step[2]));
    val1 = colorToVal(color1);
    val2 = colorToVal(color2);
    N[2] = val1 - val2;
    n_bg_borders += detectAdjacentBackground(val1, val2);

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

ISO_CATEGORICAL_SNIPPETS = dict(
    before_loop="""
        float phi_mod = 0.6180339887498948482;  // phi - 1
        float value = 0.0;
        vec4 color3 = vec4(0.0);  // final color
        vec3 dstep = 1.5 / u_shape;  // step to sample derivative, set to match iso shader
        gl_FragColor = vec4(0.0);
        bool discard_fragment = true;
        """,
    in_loop="""
        // check if value is different from the background value
        if ( floatNotEqual(val, categorical_bg_value) ) {
            // Take the last interval in smaller steps
            vec3 iloc = loc - step;
            for (int i=0; i<10; i++) {
                color = $get_data(iloc);
                color = applyColormap(color.g);
                if (floatNotEqual(color.a, 0) ) {
                    // when the value mapped to non-transparent color is reached
                    // calculate the color (apply lighting effects)
                    color = calculateCategoricalColor(color, iloc, dstep);
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
    after_loop="""
        if (discard_fragment)
            discard;
        """,
)

shaders = BaseVolume._shaders.copy()
before, after = shaders['fragment'].split('void main()')
shaders['fragment'] = before + FUNCTION_DEFINITIONS + 'void main()' + after

rendering_methods = BaseVolume._rendering_methods.copy()
rendering_methods['iso_categorical'] = ISO_CATEGORICAL_SNIPPETS


class Volume(BaseVolume):
    # add the new rendering method to the snippets dict
    _shaders = shaders
    _rendering_methods = rendering_methods
