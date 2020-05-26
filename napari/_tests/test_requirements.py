import os
import napari
import ast


def test_bundled_requirements():
    napari_root = os.path.dirname(napari.__file__)
    req_txt = os.path.join(napari_root, '..', 'requirements', 'default.txt')
    pyprj = os.path.join(napari_root, '..', 'pyproject.toml')
    with open(req_txt) as f:
        reqs = [l.split("#")[0].strip() for l in f if not l.startswith("#")]
    with open(pyprj) as f:
        content = f.read().split("[tool.briefcase.app.napari]")[1]
        content = content.split("requires = ")[1].split("]\n")[0] + "]"
        tomlreqs = ast.literal_eval(content)
    print(reqs)
    print(tomlreqs)
