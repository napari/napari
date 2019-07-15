from IPython.core.interactiveshell import InteractiveShell


CODE = '''from skimage import filters
output = filters.gaussian(viewer.layers[0].data, sigma=10)
viewer.add_image(output)
'''


def call_ipython_code(code=CODE):
    # get current running instance or create new instance
    shell = InteractiveShell.instance()
    # note: shell.extract_input_lines will return existing input lines.
    # we should save them and restore them after running our command
    # shell.set_next_input
    # shell.set_next_input(code)
    shell.pt_app.app.current_buffer.insert_text(code)
    shell.pt_app.app.current_buffer.validate_and_handle()
