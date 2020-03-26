"""
These convenience functions will be useful for searching pypi for packages
that match the plugin naming convention, and retrieving related metadata.
"""
import logging
import os
import pkgutil
import re
import sys
from subprocess import run
from typing import Dict, List, Tuple, Union
from urllib import request

import pip._internal.legacy_resolve
from pip._internal.cli.main import main as _main

from ..utils.appdirs import user_plugin_dir, user_site_packages
from ..utils.misc import absolute_resource
from .. import resources

PYPI_SIMPLE_API_URL = 'https://pypi.org/simple/'
URL_CACHE = {}  # {name: url} for packages at pypi.org/simple
VERSION_CACHE = {}  # {name: tuple of versions} for packages at pypi.org/simple


def clear_cache():
    global URL_CACHE
    global VERSION_CACHE

    URL_CACHE = {}
    VERSION_CACHE = {}


def get_packages_by_prefix(prefix: str) -> Dict[str, str]:
    """Search for packages starting with ``prefix`` on pypi.

    Packages using naming convention: http://bit.ly/pynaming-convention
    can be autodiscovered on pypi using the SIMPLE API:
    https://www.python.org/dev/peps/pep-0503/

    Returns
    -------
    dict
        {name: url} for all packages at pypi that start with ``prefix``
    """

    with request.urlopen(PYPI_SIMPLE_API_URL) as response:
        html = response.read()

    pattern = f'<a href="/simple/(.+)">({prefix}.*)</a>'
    urls = {
        name: PYPI_SIMPLE_API_URL + url
        for url, name in re.findall(pattern, html.decode())
    }
    URL_CACHE.update(urls)
    return urls


def get_package_versions(name: str) -> Tuple[str]:
    """Get available versions of a package on pypi

    Parameters
    ----------
    name : str
        name of the package

    Returns
    -------
    tuple
        versions available on pypi
    """
    url = URL_CACHE.get(name, PYPI_SIMPLE_API_URL + name)
    with request.urlopen(url) as response:
        html = response.read()

    versions = tuple(set(re.findall(f'>{name}-(.+).tar', html.decode())))
    VERSION_CACHE[name] = versions
    return versions


def install_pypi_plugin(name_or_names: Union[str, List[str]]) -> List[str]:
    names = (
        [name_or_names] if isinstance(name_or_names, str) else name_or_names
    )
    if getattr(sys, 'frozen', False):
        result = _frozen_install(names)
    else:
        cmd = ['pip', 'install']
        result = run(cmd + names, capture_output=True)
    result.check_returncode()  # if errors: raise CalledProcessError
    output = result.stdout.decode()
    for line in reversed(output.splitlines()):
        if 'Successfully installed' in line:
            return [
                i for i in line.replace('Successfully installed', '').split()
            ]
    return []


_real = pip._internal.legacy_resolve.Resolver.get_installation_order


to_install = None


class ItsOK(Exception):
    pass


def get_installs(self, req_set):
    global to_install
    to_install = [
        (i.req.name, str(i.req.specifier)) for i in _real(self, req_set)
    ]
    raise ItsOK("Done!")


def install_dry_run(pkgs, prefix="/Users/talley/Desktop/test"):

    pkg_list = [pkgs] if isinstance(pkgs, str) else pkgs

    pip._internal.legacy_resolve.Resolver.get_installation_order = get_installs
    logging.getLogger("pip").propagate = False

    cmd = ["install"]
    if prefix:
        cmd += ["--prefix", prefix]
    cmd += pkg_list
    try:
        _main(cmd)
        # TODO: why doesn't _main raise the ItsOK error?
    finally:
        pip._internal.legacy_resolve.Resolver.get_installation_order = _real
        logging.getLogger("pip").propagate = True
    return to_install


def _frozen_install(names: List[str]):
    recs = install_dry_run(names)
    env = os.environ.copy()
    cmd = [
        os.path.join(sys.exec_prefix, '_pip'),
        'install',
        '--prefix',
        '--no-dependencies',
        user_plugin_dir(),
    ]
    cmd.extend(recs)
    env['PYTHONPATH'] = user_site_packages()
    return run(cmd + names, capture_output=True)


def frozen_packages():
    # these are PACKAGE NAMES not DISTRIBUTION NAMES
    toc = set()
    importers = pkgutil.iter_importers()
    for i in importers:
        if hasattr(i, 'toc'):
            toc |= i.toc
    return {i.split('.')[0] for i in toc}


def get_bundled_dists() -> List[str]:
    res_dir = absolute_resource(os.path.dirname(resources.__file__))
    pkgs_file = os.path.join(res_dir, '_included_distributions.txt')
    if os.path.exists(pkgs_file):
        with open(pkgs_file) as f:
            return f.read().splitlines()
    return []
