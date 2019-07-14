from IPython.core.interactiveshell import InteractiveShell


def call_ipython_code(code='print("hello world")'):
    # get current running instance or create new instance
    shell = InteractiveShell.instance()
    # note: shell.extract_input_lines will return existing input lines.
    # we should save them and restore them after running our command
    # shell.set_next_input
    shell.set_next_input(code)
    shell.history_manager.store_inputs(shell.execution_count + 1, code)
    shell.ex(code)
    shell.execution_count += 1
