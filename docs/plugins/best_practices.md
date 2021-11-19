# Best practices when developing napari plugins

There are a number of good and bad practices that may not be immediately obvious
when developing a plugin.  This page

## Do

- **Update your README to make it immediately clear what your plugin does in a
  few sentences** 

    This is your elevator pitch. You'll loose a lot of potential users quickly
    if they have to look in the source code, or look through a large
    documentation page, just to know what the gist of the plugin is.

- **Use images, gifs, or movies to *show* what your plugin does**. 

  Images and movies are *extremely* useful for conveying the usefulness of your
  plugin. Both github and napari hub make it easy to embed videos in your
  readme. For example, if you have a dock widget plugin that plots something as
  the user interacts with the current layer, capture a screengrab and put it in
  your readme. Keep the `.gif` file small, as .gif files larger than 8 MB will not be played on your page on pypi.org.

  TODO: screengrab tips and code examples. 


## Don't


- **Don't explicitly declare either `PySide2` or `PyQt5` in your plugin's
  `install_requires` section.**

    *This is important!*

    Napari supports *both* PyQt and PySide backends for Qt.  It is up to the
    end-user to choose which one they want. If they installed napari with `pip
    install napari[all]`, then the `[all]` extra will (currently) install
    `PyQt5` for them from pypi.  If they installed via `conda install napari`,
    then they'll have `PyQt5`, but via anaconda cloud instead of pypi. Lastly,
    they may have installed napari with PySide2

    Here's what can go wrong if you *also* declare one of these backends:

    - If they installed via `conda install napari` and then they install your
      plugin via `pip` (or via the builtin plugin installer, which currently
      uses `pip`), then there *will* be a binary incompatibility between the
      their conda `pyqt` installation, and the new pip "`PyQt5`" installation.
      *This will very likely lead to a broken environment*. This is an
      unfortunate consequence of [package naming
      decisions](https://github.com/ContinuumIO/anaconda-issues/issues/1554),
      and it's not something napari can fix.
    - Alternatively, they may end up with *both* PyQt and PySide in their
      environement, and while that's not always an environment "death sentence",
      it too can lead to unexpected and difficulty to debug problems.

- **Don't import from PyQt5 or PySide2 in your plugin: use `qtpy`.**

    If you use `from PyQt5 import QtCore` (or similar) in your plugin, but the
    end-user has chosen to use `PySide2` for their Qt backend — or vice versa —
    then your plugin will fail to import.  Instead use `from qtpy import
    QtCore`.  `qtpy` is a [Qt compatibility
    layer](https://github.com/spyder-ide/qtpy) that will import from whatever
    backend is installed in the environment.

- **Try not to depend on packages that require C compilation, but do not offer "wheels".**

    ````{tip}
    This requires some awareness of how your dependencies are built and distributed...

    Some python packages write a portion of their code in lower level languages like C or C++ and compile that code into "C Extensions" that can be called by python at runtime.  This can *greatly* improve performance, but it means that the package must be compiled for *each* platform (i.e. Windows, Mac, Linux) that the package wants to support.  Some packages do this compilation step ahead of time, by distributing "[wheels](https://realpython.com/python-wheels/)"
    on [PyPI](https://pypi.org/)... or by providing pre-compiled packages via `conda`.  Other packages simply distribute the source code (as an "sdist") and expect the end-user to compile it on their own computer.  Compiling C code requires software that is not always installed on every computer. (If you've ever tried to `pip install` a package and had it fail with a big wall of red text saying something about `gcc`, then you've run into a package that doesn't distribute wheels, and you didn't have the software required to compile it).
    ````

    As a plugin developer, if you depend on a package that uses C extensions but doesn't distribute a pre-compiled wheel, then it's very likely that your users will run into difficulties installing your plugin:

    - *What is a "wheel"?*

      **tldr;** A wheel is a *built distribution*, containing code that is pre-compiled for a specific operating system.

      Detailed background: [What Are Python Wheels and Why Should You Care?](https://realpython.com/python-wheels/)

    - *How do I know if my dependency offers a wheel*

      There are many ways, but a sure-fire way to know is to go to the respective
      package on PyPI, and click on the "Download Files" link.  If the package offers wheels, you'll see one or more files ending in `.whl`.  For example, [napari offers a wheel](https://pypi.org/project/napari/#files).  If a package *doesn't* offer a wheel, it may still be ok if it's just a pure python package that doesn't have any C extensions...

    - *How do I know if one of my dependencies uses C Extensions?*
      
      There's no one way, but more often than not, if a package uses C extensions,  the `setup()` function in their `setup.py` file will use the [`ext_modules` argument](https://docs.python.org/3/distutils/setupscript.html#describing-extension-modules).  (for example, [see here in pytorch](https://github.com/pytorch/pytorch/blob/master/setup.py#L914))

    ````{admonition} What about conda?
    **conda** also distributes & installs pre-compiled packages, though they aren't wheels.  While this definitely a fine way to install binary dependencies in a reliable way, the built-in napari plugin installer doesn't currently work with conda.  If your dependency is only available on conda, but does not offer wheels,you *may* guide your users in using conda to install your package or one
    of your dependencies.  Just know that it may not work with the built-in plugin installer. 
    ````

- **Don't import heavy dependencies at the top of your module.**

    ````{note}
    This point will be less relevant when we move to the second generation
    [manifest-based plugin
    declaration](https://github.com/napari/napari/issues/3115), but it's still a 
    good idea to delay importing your plugin-specific dependencies and modules until *after* your hookspec has been called.  This helps napari stay quick and responsive
    at startup.  
    ````

    Consider the following example plugin:

    ```ini
    [options.entry_points]
    napari.plugion =
      plugin-name = mypackage.napari_plugin
    ```

    In this example, `my_heavy_dependency_like_tensorflow` will be imported
    *immediately* when napari is launched, and we search the entry_point
    `mypackage.napari_plugin` for decorated hook specifications.

    ```py
    # mypackage/napari_plugin.py
    from napari_plugin_engine import napari_hook_specification
    from qtpy.QtWidgets import QWidget
    from my_heavy_dependency_like_tensorflow import something_amazing

    class MyWidget(QWidget):
        def do_something_amazing(self):
            return something_amazing()

    @napari_hook_specification
    def napari_experimental_provide_dock_widget():
        return MyWidget
    ```

    This can deterioate the end-user experience, and make napari feel slugish. Best practice is to delay heavy imports until right before they are used.  The following slight modification will help napari load much faster:

    ```py
    # mypackage/napari_plugin.py
    from napari_plugin_engine import napari_hook_specification
    from qtpy.QtWidgets import QWidget

    class MyWidget(QWidget):
        def do_something_amazing(self):
            # import has been moved here, will happen only after the user
            # has opened and used this widget.
            from my_heavy_dependency_like_tensorflow import something_amazing

            return something_amazing()
    ```

    (again, the second gen napari plugin engine will help improve this situation,
    but it's still a good idea!)

- **Don't leave resources open**

    It's always good practice to clean up resources like open file handles and
    databases.  As a napari plugin it's particularly important to do this (and
    especially for Windows users).  If someone tries to use the built-in plugin manager
    to *uninstall* your plugin, open file handles and resources may cause the
    process to fail or even leave your plugin in an "installed-but-unuseable"
    state.
    
    Don't do this:

    ```py
    # my_plugin/module.py
    import json

    data_file = open("some_data_in_my_plugin.json")
    data = json.load(data_file)
    ```

    Instead, make sure to close your resource after grabbing the data:
    ```py
    with open("some_data_in_my_plugin.json") as data_file:
        data = json.load(data_file)
    ```

