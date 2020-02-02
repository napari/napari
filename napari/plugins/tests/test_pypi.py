from unittest import mock
from napari.plugins import pypi


# This method will be used by the mock to replace requests.get
def mocked_pypi_api(*args, **kwargs):
    class MockResponse:
        def __init__(self, text, status_code):
            self.text = text
            self.status_code = status_code

        def raise_for_status(self):
            return

    if args[0] == pypi.PYPI_SIMPLE_API_URL:
        text = (
            '<!DOCTYPE html>\n<html>\n  <head>\n  <title>Simple index</title>'
            '\n </head>\n  <body>\n  <a href="/simple/package1/">package1</a>'
            '\n  <a href="/simple/package2/">packge2</a>\n   </body>\n</html>'
        )
        return MockResponse(text, 200)
    elif args[0].startswith(pypi.PYPI_SIMPLE_API_URL):
        name = args[0].split(pypi.PYPI_SIMPLE_API_URL)[1]
        text = (
            f'<!DOCTYPE html>\n<html>\n  <head>\n    <title>Links for {name}'
            f'</title>\n  </head>\n  <body>\n    <h1>Links for {name}</h1>\n'
            f'<a href="http://pythonhosted.org/{name}-0.1.0.tar.gz#sha256=7">'
            f'{name}-0.1.0.tar.gz</a><br/>\n </body>\n</html>'
        )
        return MockResponse(text, 200)
    return MockResponse(None, 404)


@mock.patch('napari.plugins.pypi.requests.get', side_effect=mocked_pypi_api)
def test_get_packages_by_prefix(mock_get):
    urls = pypi.get_packages_by_prefix('package')
    assert 'package1' in urls
    assert urls['package1'] == 'https://pypi.org/simple/package1/'


@mock.patch('napari.plugins.pypi.requests.get', side_effect=mocked_pypi_api)
def test_get_package_versions(mock_get):
    versions = pypi.get_package_versions('package')
    assert '0.1.0' in versions
