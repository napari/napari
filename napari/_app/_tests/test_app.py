from napari._app import app
from napari.layers import Points


def test_app():
    """just make sure our app model is registering menus and commands"""
    assert app.name == 'napari'
    assert list(app.menus)
    assert list(app.commands)
    # assert list(app.keybindings)  # don't have any yet


def test_app_injection():
    """Simple test to make sure napari namespaces are working in app injection."""

    def use_points(points: 'Points'):
        return points

    p = Points()

    def provide_points() -> 'Points':
        return p

    with app.injection_store.register(providers=[(provide_points,)]):
        assert app.injection_store.inject(use_points)() is p
