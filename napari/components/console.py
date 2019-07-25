from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager


def make_console(user_ns):
    global ipython_widget  # Prevent from being garbage collected

    # Create an in-process kernel
    kernel_manager = QtInProcessKernelManager()
    kernel_manager.start_kernel(show_banner=False)
    kernel = kernel_manager.kernel
    kernel.gui = 'qt'
    kernel.shell.user_ns.update(user_ns)

    kernel_client = kernel_manager.client()
    kernel_client.start_channels()

    ipython_widget = RichJupyterWidget()
    ipython_widget.kernel_manager = kernel_manager
    ipython_widget.kernel_client = kernel_client

    ipython_widget.setStyleSheet(
        """QPlainTextEdit, QTextEdit {
        background-color: black;
        background-clip: padding;
        color: white;
        selection-background-color: #ccc;
    }
    .inverted {
        background-color: white;
        color: black;
    }
    .error { color: red; }
    .in-prompt-number { font-weight: bold; }
    .out-prompt-number { font-weight: bold; }
    .in-prompt { color: green; }
    .out-prompt { color: darkred; }
    """
    )

    return ipython_widget


#
# CODE = '''from skimage import filters
# output = filters.gaussian(viewer.layers[0].data, sigma=10)
# viewer.add_image(output)
# '''
#
#
# def call_ipython_code(code=CODE):
#     # get current running instance or create new instance
#     shell = get_ipython()
#     # note: shell.extract_input_lines will return existing input lines.
#     # we should save them and restore them after running our command
#     if shell is None:  # pure-python interpreter
#         pass
#     elif hasattr(shell, 'pt_app'):  # ipython terminal
#         shell.pt_app.app.current_buffer.insert_text(code)
#         shell.pt_app.app.current_buffer.validate_and_handle()
#     else:  # ipython kernel
#         shell.kernel._publish_execute_input(
#             code, None, shell.kernel.execution_count
#         )
#         shell.run_cell(code)
