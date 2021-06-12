from unittest import mock

from napari.plugins import pypi


class FakeResponse:
    def __init__(self, *, data: bytes):
        self.data = data

    def read(self):
        return self.data

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return


txtA = (
    b'<!DOCTYPE html>\n<html>\n  <head>\n  <title>Simple index</title>'
    b'\n </head>\n  <body>\n  <a href="/simple/package1/">package1</a>'
    b'\n  <a href="/simple/package2/">packge2</a>\n   </body>\n</html>'
)

txtB = (
    b'<!DOCTYPE html>\n<html>\n  <head>\n    <title>Links for package'
    b'</title>\n  </head>\n  <body>\n    <h1>Links for package</h1>\n'
    b'<a href="http://pythonhosted.org/package-0.1.0.tar.gz#sha256=7">'
    b'package-0.1.0.tar.gz</a><br/>\n </body>\n</html>'
)


@mock.patch(
    'napari.plugins.pypi.urlopen', return_value=FakeResponse(data=txtA)
)
def test_get_packages_by_prefix(mock_get):
    urls = pypi.get_packages_by_prefix('package')
    assert 'package1' in urls
    assert urls['package1'] == 'https://pypi.org/simple/package1/'


@mock.patch(
    'napari.plugins.pypi.urlopen', return_value=FakeResponse(data=txtB)
)
def test_get_package_versions(mock_get):
    versions = pypi.get_package_versions('package')
    assert '0.1.0' in versions
