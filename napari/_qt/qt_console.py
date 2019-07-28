from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.manager import QtKernelManager
from qtconsole.client import QtKernelClient
from IPython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from ipykernel.zmqshell import ZMQInteractiveShell
from ipykernel.connect import get_connection_file


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
