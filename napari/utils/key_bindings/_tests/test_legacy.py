import pytest
from app_model import Application

from napari.utils.key_bindings.legacy import (
    KeymapProvider,
)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_bind_provider(_mock_app: Application):
    class Foo(KeymapProvider):
        @classmethod
        def _type_string(cls):
            return 'image'

    @Foo.bind_key('Control-A')
    def abc(image: Foo):
        pass

    command_id = f'autogen:{abc.__qualname__}'

    assert command_id in _mock_app.commands
    assert _mock_app.keybindings.get_keybinding(command_id).when is not None
