from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.manager import QtKernelManager
from qtconsole.client import QtKernelClient
from IPython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.lib.kernel import find_connection_file
from ipykernel.inprocess.ipkernel import InProcessInteractiveShell
from ipykernel.zmqshell import ZMQInteractiveShell
from ipykernel.connect import get_connection_file


class QtConsole(RichJupyterWidget):
    """Qt view for console.

    Parameters
    ----------
    user_variables : dict
        Dictionary of user variables to declare in console name space.

    Attributes
    ----------
    kernel_client : qtconsole.inprocess.QtInProcessKernelClient,
                    qtconsole.client.QtKernelClient, or None
        Client for the kernel if it exists, None otherwise.
    shell : ipykernel.inprocess.ipkernel.InProcessInteractiveShell,
            ipykernel.zmqshell.ZMQInteractiveShell, or None.
        Shell for the kernel if it exists, None otherwise.
    """

    def __init__(self, user_variables=None):
        super().__init__()

        # get current running instance or create new instance
        shell = get_ipython()

        if shell is None:
            # If there is no currently running instance create an in-process
            # kernel.
            kernel_manager = QtInProcessKernelManager()
            kernel_manager.start_kernel(show_banner=False)
            kernel_manager.kernel.gui = 'qt'

            kernel_client = kernel_manager.client()
            kernel_client.start_channels()

            self.kernel_manager = kernel_manager
            self.kernel_client = kernel_client
            self.shell = kernel_manager.kernel.shell
            self.push = self.shell.push
        elif type(shell) == InProcessInteractiveShell:
            # If there is an existing running InProcessInteractiveShell
            # it is likely because multiple viewers have been launched from
            # the same process. In that case create a new kernel.
            kernel_manager = QtInProcessKernelManager()
            kernel_manager.start_kernel(show_banner=False)
            kernel_manager.kernel.gui = 'qt'

            kernel_client = kernel_manager.client()
            kernel_client.start_channels()

            self.kernel_manager = kernel_manager
            self.kernel_client = kernel_client
            self.shell = kernel_manager.kernel.shell
            self.push = self.shell.push
        elif isinstance(shell, TerminalInteractiveShell):
            # if launching from an ipython terminal then adding a console is
            # not supported. Instead users should use the ipython terminal for
            # the same functionality.
            self.kernel_client = None
            self.kernel_manager = None
            self.shell = None
            self.push = lambda var: None

        elif isinstance(shell, ZMQInteractiveShell):
            # if launching from jupyter notebook, connect to the existing
            # kernel
            kernel_client = QtKernelClient(
                connection_file=get_connection_file()
            )
            kernel_client.load_connection_file()
            kernel_client.start_channels()

            self.kernel_manager = None
            self.kernel_client = kernel_client
            self.shell = shell
            self.push = self.shell.push
        else:
            raise ValueError(
                'ipython shell not recognized; ' f'got {type(shell)}'
            )

        # Add any user variables
        user_variables = user_variables or {}
        self.push(user_variables)

        self.enable_calltips = False

        # TODO: Try to get console from jupyter to run without a shift click
        # self.execute_on_complete_input = True

    def shutdown(self):
        if self.kernel_client is not None:
            self.kernel_client.stop_channels()
        if self.kernel_manager is not None:
            self.kernel_manager.shutdown_kernel()
