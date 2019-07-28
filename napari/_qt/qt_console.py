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
from ..util.theme import template

print(list(get_all_styles()))


class QtConsole(RichJupyterWidget):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer

        # self.connect.closeEvent(self.shutdown_kernel)
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

        self.enable_calltips = False
        # Try to get console from jupyter to run without a shift click
        # self.execute_on_complete_input = True

    def update_palette(self, palette):
        raw_stylesheet = """QPlainTextEdit, QTextEdit {
                    background-color: {{ foreground }};
                    background-clip: padding;
                    color: {{ text }};
                    selection-background-color: {{ highlight }};
                    margin: 10px;
                }
                .inverted {
                    background-color: {{ background }};
                    color: {{ foreground }};
                }
                .error { color: #b72121; }
                .in-prompt-number { font-weight: bold; }
                .out-prompt-number { font-weight: bold; }
                .in-prompt { color: #6ab825; }
                .out-prompt { color: #b72121; }
                """
        themed_stylesheet = template(raw_stylesheet, **palette)
        self.syntax_style = palette['syntax_style']
        self.style_sheet = themed_stylesheet
