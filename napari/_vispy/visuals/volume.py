from vispy.scene.visuals import Volume as BaseVolume
from vispy.visuals.shaders import Function
from vispy.visuals.volume import frag_dict

ISO_CATEGORICAL_SNIPPETS = dict(
    before_loop="""
        vec4 color3 = vec4(0.0);  // final color
        vec3 dstep = 1.5 / u_shape;  // step to sample derivative, set to match iso shader
        gl_FragColor = vec4(0.0);
        """,
    in_loop="""
        // the tolerance for testing equality of floats with floatEqual and floatNotEqual
        const float u_equality_tolerance = 1e-8;

        // the background value for the iso_categorical shader
        const float u_categorical_bg_value = 0;

        bool floatNotEqual(float val1, float val2, float equality_tolerance)
        {
            // check if val1 and val2 are not equal
            bool not_equal = abs(val1 - val2) > equality_tolerance;

            return not_equal;
        }

        bool floatEqual(float val1, float val2, float equality_tolerance)
        {
            // check if val1 and val2 are equal
            bool equal = abs(val1 - val2) < equality_tolerance;

            return equal;
        }


        int detectAdjacentBackground(float val_neg, float val_pos)
        {
            // determine if the adjacent voxels along an axis are both background
            int adjacent_bg = int( floatEqual(val_neg, u_categorical_bg_value, u_equality_tolerance) );
            adjacent_bg = adjacent_bg * int( floatEqual(val_pos, u_categorical_bg_value, u_equality_tolerance) );
            return adjacent_bg;
        }

        vec4 calculateCategoricalColor(vec4 betterColor, vec3 loc, vec3 step)
        {
            // Calculate color by incorporating ambient and diffuse lighting
            vec4 color0 = $sample(u_volumetex, loc);
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
            color1 = $sample( u_volumetex, loc+vec3(-step[0],0.0,0.0) );
            color2 = $sample( u_volumetex, loc+vec3(step[0],0.0,0.0) );
            val1 = colorToVal(color1);
            val2 = colorToVal(color2);
            N[0] = val1 - val2;
            n_bg_borders += detectAdjacentBackground(val1, val2);

            color1 = $sample( u_volumetex, loc+vec3(0.0,-step[1],0.0) );
            color2 = $sample( u_volumetex, loc+vec3(0.0,step[1],0.0) );
            val1 = colorToVal(color1);
            val2 = colorToVal(color2);
            N[1] = val1 - val2;
            n_bg_borders += detectAdjacentBackground(val1, val2);

            color1 = $sample( u_volumetex, loc+vec3(0.0,0.0,-step[2]) );
            color2 = $sample( u_volumetex, loc+vec3(0.0,0.0,step[2]) );
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
        // check if value is different from the background value
        if ( floatNotEqual(val, u_categorical_bg_value, u_equality_tolerance) ) {
            // Take the last interval in smaller steps
            vec3 iloc = loc - step;
            for (int i=0; i<10; i++) {
                color = $sample(u_volumetex, iloc);
                if (floatNotEqual(color.g, u_categorical_bg_value, u_equality_tolerance) ) {
                    // when the non-background value is reached
                    // calculate the color (apply lighting effects)
                    color = applyColormap(color.g);
                    color = calculateCategoricalColor(color, iloc, dstep);
                    gl_FragColor = color;

                    // set the variables for the depth buffer
                    surface_point = iloc * u_shape;
                    surface_found = true;

                    iter = nsteps;
                    break;
                }
                iloc += step * 0.1;
            }
        }
        """,
    after_loop="""
        if (surface_found == false) {
            discard;
        }
        """,
)


frag_dict['iso_categorical'] = ISO_CATEGORICAL_SNIPPETS


class Volume(BaseVolume):
    # override these methods to use our frag_dict
    @property
    def _before_loop_snippet(self):
        return frag_dict[self.method]['before_loop']

    @property
    def _in_loop_snippet(self):
        return frag_dict[self.method]['in_loop']

    @property
    def _after_loop_snippet(self):
        return frag_dict[self.method]['after_loop']

    @property
    def method(self):
        return super().method

    @method.setter
    def method(self, method):
        # Check and save
        known_methods = list(frag_dict.keys())
        if method not in known_methods:
            raise ValueError(
                'Volume render method should be in %r, not %r'
                % (known_methods, method)
            )
        self._method = method
        # Get rid of specific variables - they may become invalid
        if 'u_threshold' in self.shared_program:
            self.shared_program['u_threshold'] = None
        if 'u_attenuation' in self.shared_program:
            self.shared_program['u_attenuation'] = None

        # $sample needs to be unset and re-set, since it's present inside the snippets.
        #       Program should probably be able to do this automatically
        self.shared_program.frag['sample'] = None
        self.shared_program.frag[
            'raycasting_setup'
        ] = self._raycasting_setup_snippet
        self.shared_program.frag['before_loop'] = self._before_loop_snippet
        self.shared_program.frag['in_loop'] = self._in_loop_snippet
        self.shared_program.frag['after_loop'] = self._after_loop_snippet
        self.shared_program.frag[
            'sampler_type'
        ] = self._texture.glsl_sampler_type
        self.shared_program.frag['sample'] = self._texture.glsl_sample
        self.shared_program.frag['cmap'] = Function(self._cmap.glsl_map)
        self.shared_program['texture2D_LUT'] = self.cmap.texture_lut()
        self.update()
