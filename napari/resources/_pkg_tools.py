import os
import re
import sys
from importlib import import_module
import napari

import pkg_resources

ROOT = os.path.dirname(os.path.dirname(napari.__file__))


def dist2modules(dist):
    if isinstance(dist, str):
        dist = pkg_resources.get_distribution(dist)
    if "top_level.txt" in dist.metadata_listdir(""):
        return list(dist.get_metadata_lines("top_level.txt"))
    return []


def get_packages(imports=()):
    for mod in imports:
        import_module(mod)
    top_modules = {
        m.split(".")[0] for m in sys.modules if not m.startswith("_")
    }
    top_modules -= set(sys.builtin_module_names)
    packages = set()
    for dist in pkg_resources.working_set:
        for module_name in dist2modules(dist):
            if module_name in top_modules:
                packages.add(str(dist.as_requirement()))
    return packages


def write_included_packages(outfile=None, imports=()):
    outfile = outfile or os.path.join(
        os.path.dirname(__file__), '_included_distributions.txt'
    )
    if not imports:
        imports = ['napari', 'pip', 'setuptools']
        with open(os.path.join(ROOT, 'requirements', 'default.txt')) as f:
            for line in f.read().splitlines():
                if not line.startswith("#"):
                    mods = dist2modules(re.split(r'[=><\[]', line)[0])
                    imports.extend(mods)

    packages = get_packages(imports)
    with open(outfile, "w") as file:
        file.write("\n".join(sorted(packages)))


if __name__ == "__main__":
    write_included_packages()
