from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.manager import QtKernelManager
from qtconsole.client import QtKernelClient
from IPython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell
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

        user_variables = user_variables or {}

        # get current running instance or create new instance
        shell = get_ipython()

        if shell is None or type(shell) == InProcessInteractiveShell:
            # If there is no currently running instance create an in-process
            # kernel or if there is an old running InProcessInteractiveShell
            # then just create a new one - necessary for our tests to pass

            kernel_manager = QtInProcessKernelManager()
            kernel_manager.start_kernel(show_banner=False)
            kernel_manager.kernel.gui = 'qt'
            kernel_manager.kernel.shell.push(user_variables)

            kernel_client = kernel_manager.client()
            kernel_client.start_channels()

            self.kernel_client = kernel_client
            self.shell = kernel_manager.kernel.shell

        elif isinstance(shell, TerminalInteractiveShell):
            # if launching from an ipython terminal then adding a console is
            # not supported. Instead users should use the ipython terminal for
            # the same functionality.
            self.kernel_client = None
            self.shell = None

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
            self.shell.push(user_variables)
        else:
            raise ValueError(
                'ipython shell not recognized; ' f'got {type(shell)}'
            )

        self.enable_calltips = False

        # TODO: Try to get console from jupyter to run without a shift click
        # self.execute_on_complete_input = True
