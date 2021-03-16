import numpy as np
from vispy.color import Color, get_colormap
from vispy.gloo.buffer import VertexBuffer
from vispy.scene.visuals import create_visual_node
from vispy.visuals.visual import Visual

from .vendored import MeshVisual as BaseMeshVisual

# Unified shader
shader_template = """
{setup}
void main() {{
    {main_function}
}}
"""

standard_vertex_snippets = dict(
    setup="""
    varying vec4 v_base_color;
    """,
    main_function="""
    v_base_color = $color_transform($base_color);
    gl_Position = $transform($to_vec4($position));
    """,
)

standard_vertex_shader = shader_template.format(**standard_vertex_snippets)

shaded_vertex_snippets = dict(
    setup="""
    varying vec3 v_normal_vec;
    varying vec3 v_light_vec;
    varying vec3 v_eye_vec;
    varying vec4 v_ambientk;
    varying vec4 v_light_color;
    varying vec4 v_base_color;
    """,
    main_function="""
    v_ambientk = $ambientk;
    v_light_color = $light_color;
    v_base_color = $color_transform($base_color);
    vec4 pos_scene = $visual2scene($to_vec4($position));
    vec4 normal_scene = $visual2scene(vec4($normal, 1.0));
    vec4 origin_scene = $visual2scene(vec4(0.0, 0.0, 0.0, 1.0));
    normal_scene /= normal_scene.w;
    origin_scene /= origin_scene.w;
    vec3 normal = normalize(normal_scene.xyz - origin_scene.xyz);
    v_normal_vec = normal; //VARYING COPY
    vec4 pos_front = $scene2doc(pos_scene);
    pos_front.z += 0.01;
    pos_front = $doc2scene(pos_front);
    pos_front /= pos_front.w;
    vec4 pos_back = $scene2doc(pos_scene);
    pos_back.z -= 0.01;
    pos_back = $doc2scene(pos_back);
    pos_back /= pos_back.w;
    vec3 eye = normalize(pos_front.xyz - pos_back.xyz);
    v_eye_vec = eye; //VARYING COPY
    vec3 light = normalize($light_dir.xyz);
    v_light_vec = light; //VARYING COPY
    gl_Position = $transform($to_vec4($position));
    """,
)

shaded_vertex_shader = shader_template.format(**shaded_vertex_snippets)

standard_fragment_snippets = dict(
    setup="""
    varying vec4 v_base_color;
    """,
    main_function="""
    gl_FragColor = v_base_color;
    """,
)

standard_fragment_shader = shader_template.format(**standard_fragment_snippets)

shaded_fragment_snippets = dict(
    setup="""
    varying vec3 v_normal_vec;
    varying vec3 v_light_vec;
    varying vec3 v_eye_vec;
    varying vec4 v_ambientk;
    varying vec4 v_light_color;
    varying vec4 v_base_color;
    """,
    main_function="""
    //DIFFUSE
    float diffusek = dot(v_light_vec, v_normal_vec);
    // clamp, because 0 < theta < pi/2
    diffusek  = clamp(diffusek, 0.0, 1.0);
    vec4 diffuse_color = v_light_color * diffusek;
    //SPECULAR
    //reflect light wrt normal for the reflected ray, then
    //find the angle made with the eye
    float speculark = 0.0;
    if ($shininess > 0.) {
        speculark = dot(reflect(v_light_vec, v_normal_vec), v_eye_vec);
        speculark = clamp(speculark, 0.0, 1.0);
        //raise to the material's shininess, multiply with a
        //small factor for spread
        speculark = 20.0 * pow(speculark, 1.0 / $shininess);
    }
    vec4 specular_color = v_light_color * speculark;
    gl_FragColor = v_base_color * (v_ambientk + diffuse_color) + specular_color;
    """,
)

shaded_fragment_shader = shader_template.format(**shaded_fragment_snippets)

# variables in the shaded shader but not the standard shader
shaded_only_variables = (
    'v_normal_vec',
    'v_light_vec',
    'v_eye_vec',
    'v_ambientk',
    'v_light_color',
)

shader_dict = {
    'standard': {
        'frag': standard_fragment_shader,
        'vert': standard_vertex_shader,
    },
    'shaded': {
        'frag': shaded_fragment_shader,
        'vert': shaded_vertex_shader,
    },
}


# Custom MeshVisual class to unify shader code for easier switching
# between shading modes
class MeshVisual(BaseMeshVisual):
    def __init__(
        self,
        vertices=None,
        faces=None,
        vertex_colors=None,
        face_colors=None,
        color=(0.5, 0.5, 1, 1),
        vertex_values=None,
        meshdata=None,
        shading=None,
        mode='triangles',
        **kwargs,
    ):

        # Function for computing phong shading
        # self._phong = Function(phong_template)

        # Visual.__init__ -> prepare_transforms() -> uses shading
        self.shading = shading

        Visual.__init__(self, vcode='', fcode='', **kwargs)

        self.set_gl_state('translucent', depth_test=True, cull_face=False)

        # Define buffers
        self._vertices = VertexBuffer(np.zeros((0, 3), dtype=np.float32))
        self._normals = VertexBuffer(np.zeros((0, 3), dtype=np.float32))
        self._ambient_light_color = Color((0.3, 0.3, 0.3, 1.0))
        self._light_dir = (10, 5, -5)
        self._shininess = 1.0 / 200.0
        self._cmap = get_colormap('cubehelix')
        self._clim = 'auto'

        # Uniform color
        self._color = Color(color)

        # Init
        self._bounds = None
        # Note we do not call subclass set_data -- often the signatures
        # do no match.
        MeshVisual.set_data(
            self,
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            face_colors=face_colors,
            vertex_values=vertex_values,
            meshdata=meshdata,
            color=color,
        )

        # primitive mode
        self._draw_mode = mode
        self.freeze()

    @property
    def shading(self):
        """The shading method used."""
        return self._shading

    @shading.setter
    def shading(self, shading):
        print(shading)
        known_shading_modes = (None, 'flat', 'smooth')
        if shading not in known_shading_modes:
            raise ValueError(
                f'Shading should be in {known_shading_modes}, not {shading}'
            )
        self.shader = self._shading_to_shader(shading)
        self._shading = shading
        self._update_shared_program()

    def _shading_to_shader(self, shading):
        """Infer which shader to use from shading mode"""
        standard_shading_modes = [None]
        shaded_shading_modes = ['flat', 'smooth']
        if shading in standard_shading_modes:
            shader = 'standard'
        elif shading in shaded_shading_modes:
            shader = 'shaded'
        return shader

    @property
    def shader(self):
        """The shader to use

        Current options are:

            * standard: no lighting is used in the computation of colors.
            * shaded: lighting is used in the computation of color
              (Phong shading - https://en.wikipedia.org/wiki/Phong_shading)
        """
        return self._shader

    @shader.setter
    def shader(self, shader):
        known_shaders = list(shader_dict.keys())
        if shader not in known_shaders:
            raise ValueError(
                f'Shader should be in {known_shaders}, not {shader}'
            )
        self._shader = shader

        # need to check if shared program has been created yet
        if getattr(self, 'shared_program', None):
            self._cleanup_shared_program()
            self.shared_program.frag = shader_dict[shader]['frag']
            self.shared_program.vert = shader_dict[shader]['vert']
            self.update()

    def _cleanup_shared_program(self):
        """Remove extra variables which may become invalid on shader change"""
        common_variables = [
            v for v in shaded_only_variables if v in self.shared_program
        ]
        for v in common_variables:
            self.shared_program[v] = None

    def _update_shaders(self):
        self.shared_program.frag = shader_dict[self._shader]['frag']
        self.shared_program.vert = shader_dict[self._shader]['vert']

    def _update_shared_program(self):
        if hasattr(self, 'shared_program'):
            self._cleanup_shared_program()
            self._update_shaders()
            self._prepare_transforms(self)

    @staticmethod
    def _prepare_transforms(view):
        if not hasattr(view, '_program'):
            return

        if 'transform' in view.view_program.vert.template_vars:
            tr = view.transforms.get_transform()
            view.view_program.vert['transform'] = tr  # .simplified

        if view.shading is not None:
            visual2scene = view.transforms.get_transform('visual', 'scene')
            scene2doc = view.transforms.get_transform('scene', 'document')
            doc2scene = view.transforms.get_transform('document', 'scene')
            view.shared_program.vert['visual2scene'] = visual2scene
            view.shared_program.vert['scene2doc'] = scene2doc
            view.shared_program.vert['doc2scene'] = doc2scene


Mesh = create_visual_node(MeshVisual)
