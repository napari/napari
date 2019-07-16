from IPython import get_ipython


CODE = '''from skimage import filters
output = filters.gaussian(viewer.layers[0].data, sigma=10)
viewer.add_image(output)
'''


def call_ipython_code(code=CODE):
    # get current running instance or create new instance
    shell = get_ipython()
    # note: shell.extract_input_lines will return existing input lines.
    # we should save them and restore them after running our command
    if shell is None:  # pure-python interpreter
        pass
    elif hasattr(shell, 'pt_app'):  # ipython terminal
        shell.pt_app.app.current_buffer.insert_text(code)
        shell.pt_app.app.current_buffer.validate_and_handle()
    else:  # ipython kernel
        shell.kernel._publish_execute_input(
            code, None, shell.kernel.execution_count
        )
        shell.run_cell(code)
