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

    if shell is None:
        # If there is no currently running instance create an in-process kernel
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

    elif isinstance(shell, TerminalInteractiveShell):
        # if launching from an ipython terminal then adding a console is not
        # supported. Instead users should use the ipython terminal for
        # the same functionality.
        ipython_widget = None

    elif isinstance(shell, ZMQInteractiveShell):
        # if launching from jupyter notebook, connect to the existing kernel
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
