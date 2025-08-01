from napari._app_model import get_app_model

app = get_app_model()

app.commands.execute_command("napari:window:window:toggle_window_console").result()
