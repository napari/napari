from __future__ import annotations

from typing import TYPE_CHECKING, Any

from packaging.requirements import Requirement
from tox.plugin import impl

if TYPE_CHECKING:
    from tox.tox_env.api import ToxEnv

try:
    from tomllib import load as load_toml
except ImportError:
    from tomli import load as load_toml


def explode_nested_deps(
    requirements: list[Requirement],
    optional_deps: dict[str, list[str]],
):
    napari_deps = [x for x in requirements if x.name == 'napari']
    if not napari_deps:
        return
    for dep in napari_deps:
        requirements.remove(dep)
        extras = dep.extras
        for extra in extras:
            if extra not in optional_deps:
                raise ValueError(
                    f'Extra {extra!r} not found in optional-dependencies'
                )
            for req_str in optional_deps[extra]:
                requirements.append(Requirement(req_str))
    explode_nested_deps(requirements, optional_deps)


@impl
def tox_on_install(
    tox_env: ToxEnv, arguments: Any, section: str, of_type: str
):
    if of_type == 'dependency-groups':
        pyproject_file = tox_env.core['tox_root'] / 'pyproject.toml'
        pyproject = load_toml(pyproject_file.open('rb'))
        optional_deps = pyproject['project']['optional-dependencies']
        explode_nested_deps(arguments, optional_deps)
