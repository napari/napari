from qtpy.QtWidgets import QWidget
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.manager import QtKernelManager
from qtconsole.client import QtKernelClient
from IPython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from ipykernel.zmqshell import ZMQInteractiveShell
from ipykernel.connect import get_connection_file


def make_console(user_ns):
    # Prevent from being garbage collected
    global ipython_widget

    # get current running instance or create new instance
    shell = get_ipython()

    if shell is None:  # pure-python interpreter
        # Create an in-process kernel
        kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel(show_banner=False)
        kernel_manager.kernel.gui = 'qt'
        kernel_manager.kernel.shell.user_ns.update(user_ns)

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

    elif type(shell) == TerminalInteractiveShell:  # ipython terminal
        ipython_widget = QWidget
        print(type(shell))
        # kernel_manager = QtInProcessKernelManager()
        # #kernel_manager.start_kernel(show_banner=False)
        # kernel_manager.kernel.gui = 'qt'
        # kernel_manager.kernel.shell = shell
        # kernel_manager.kernel.shell.user_ns.update(user_ns)
        #
        # kernel_client = kernel_manager.client()
        # kernel_client.start_channels()
        #
        # ipython_widget = RichJupyterWidget()
        # ipython_widget.kernel_manager = kernel_manager
        # ipython_widget.kernel_client = kernel_client

        # kernel_client = QtKernelClient(connection_file=get_connection_file())
        # kernel_client.load_connection_file()
        # kernel_client.start_channels()
        #
        # ipython_widget = RichJupyterWidget()
        # ipython_widget.kernel_client = kernel_client
        # ipython_widget.shell = shell
        # ipython_widget.shell.user_ns.update(user_ns)

    elif type(shell) == ZMQInteractiveShell:  # ipython kernel
        kernel_client = QtKernelClient(connection_file=get_connection_file())
        kernel_client.load_connection_file()
        kernel_client.start_channels()

        ipython_widget = RichJupyterWidget()
        ipython_widget.kernel_client = kernel_client
        ipython_widget.shell = shell
        ipython_widget.shell.user_ns.update(user_ns)

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
    else:
        raise ValueError('ipython shell not recognized; ' f'got {type(shell)}')

    return ipython_widget


# def make_console(user_ns):
#     """Start a kernel, connect to it, and create a RichJupyterWidget to use it
#     """
#     kernel_manager = QtKernelManager(kernel_name='python3')
#     kernel_manager.start_kernel()
#
#
#     kernel_client = kernel_manager.client()
#     kernel_client.start_channels()
#
#
#     jupyter_widget = RichJupyterWidget()
#     jupyter_widget.kernel_manager = kernel_manager
#     jupyter_widget.kernel_client = kernel_client
#
#     #kernel_manager.kernel.shell.user_ns.update(user_ns)
#     print(dir(jupyter_widget.kernel_manager.kernel))
#     print(dir(jupyter_widget.kernel_manager))
#     print(dir(jupyter_widget))
#
#     return jupyter_widget


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
