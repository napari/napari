from napari._app_model import get_app
from napari.layers import Points


def test_app():
    """just make sure our app model is registering menus and commands"""
    app = get_app()
    assert app.name == 'test_app'
    assert list(app.menus)
    assert list(app.commands)
    # assert list(app.keybindings)  # don't have any yet


def test_app_injection():
    """Simple test to make sure napari namespaces are working in app injection."""
    app = get_app()

    def use_points(points: 'Points'):
        return points

    p = Points()

    def provide_points() -> 'Points':
        return p

    with app.injection_store.register(providers=[(provide_points,)]):
        injected = app.injection_store.inject(use_points)
        assert injected() is p
