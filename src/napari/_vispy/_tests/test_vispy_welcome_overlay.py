from napari._vispy.overlays.welcome import VispyWelcomeOverlay


def _welcome_overlay(viewer) -> VispyWelcomeOverlay:
    """Return the vispy welcome overlay backend instance for a viewer."""
    return viewer.window._qt_viewer.canvas._overlay_to_visual[
        viewer.welcome_screen
    ][0]


def test_welcome_tip_is_selected_once_per_display(
    make_napari_viewer, monkeypatch
):
    viewer = make_napari_viewer(show=True, show_welcome_screen=True)
    viewer.welcome_screen.tips = ('first tip', 'second tip')
    welcome_overlay = _welcome_overlay(viewer)

    picks = iter(('first tip', 'second tip'))
    monkeypatch.setattr(
        'napari._vispy.overlays.welcome.choice',
        lambda tips: next(picks),
    )

    viewer.welcome_screen.visible = False
    viewer.welcome_screen.visible = True
    assert 'first tip' in welcome_overlay.node._tip

    viewer.welcome_screen.visible = False
    viewer.welcome_screen.visible = True
    assert 'second tip' in welcome_overlay.node._tip
    viewer.welcome_screen.visible = False


def test_welcome_tip_falls_back_to_default(make_napari_viewer):
    viewer = make_napari_viewer(show=True, show_welcome_screen=True)
    viewer.welcome_screen.tips = ()
    welcome_overlay = _welcome_overlay(viewer)

    viewer.welcome_screen.visible = False
    viewer.welcome_screen.visible = True
    assert "You're awesome!" in welcome_overlay.node._tip
    viewer.welcome_screen.visible = False
