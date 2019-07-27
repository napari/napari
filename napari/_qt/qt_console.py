from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.manager import QtKernelManager
from qtconsole.client import QtKernelClient
from qtconsole.styles import sheet_from_template
from IPython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from ipykernel.zmqshell import ZMQInteractiveShell
from ipykernel.connect import get_connection_file
from pygments.styles import get_all_styles

print(list(get_all_styles()))


class QtConsole(RichJupyterWidget):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer
        # get current running instance or create new instance
        shell = get_ipython()

        if shell is None:
            # If there is no currently running instance create an in-process
            # kernel
            kernel_manager = QtInProcessKernelManager()
            kernel_manager.start_kernel(show_banner=False)
            kernel_manager.kernel.gui = 'qt'
            kernel_manager.kernel.shell.user_ns.update({'viewer': self.viewer})

            kernel_client = kernel_manager.client()
            kernel_client.start_channels()

            self.kernel_manager = kernel_manager
            self.kernel_client = kernel_client

        elif isinstance(shell, TerminalInteractiveShell):
            # if launching from an ipython terminal then adding a console is
            # not supported. Instead users should use the ipython terminal for
            # the same functionality.
            self.kernel_client = None

        elif isinstance(shell, ZMQInteractiveShell):
            # if launching from jupyter notebook, connect to the existing
            # kernel
            kernel_client = QtKernelClient(
                connection_file=get_connection_file()
            )
            kernel_client.load_connection_file()
            kernel_client.start_channels()

            self.kernel_client = kernel_client
            self.shell = shell
            self.shell.user_ns.update({'viewer': self.viewer})
        else:
            raise ValueError(
                'ipython shell not recognized; ' f'got {type(shell)}'
            )

        # style_sheet = sheet_from_template('monokai')

        style_sheet = """QPlainTextEdit, QTextEdit {
                    background-color: black;
                    background-clip: padding;
                    color: white;
                    selection-background-color: white;
                }
                .inverted {
                    background-color: white;
                    color: black;
                }
                .error { color: red; }
                .in-prompt-number { font-weight: bold; }
                .out-prompt-number { font-weight: bold; }
                .in-prompt { color: lime; }
                .out-prompt { color: red; }
                """

        print(style_sheet)
        # self.setStyleSheet(style_sheet)
        self.style_sheet = style_sheet
