(best-practices)=

# Best practices

There are a number of good and bad practices that may not be immediately obvious
when developing a plugin.  This page covers some known practices that could
affect the ability to install or use your plugin effectively.


(best-practices-no-qt-backend)=

## Don't include `PySide2` or `PyQt5` in your plugin's dependencies.

*This is important!*

Napari supports *both* PyQt and PySide backends for Qt.  It is up to the
end-user to choose which one they want. If they installed napari with `pip
install napari[all]`, then the `[all]` extra will (currently) install `PyQt5`
for them from pypi.  If they installed via `conda install napari`, then they'll
have `PyQt5`, but via anaconda cloud instead of pypi. Lastly, they may have
installed napari with PySide2.

Here's what can go wrong if you *also* declare one of these backends in the
`install_requires` section of your plugin metadata:

- If they installed via `conda install napari` and then they install your plugin
  via `pip` (or via the builtin plugin installer, which currently uses `pip`),
  then there *will* be a binary incompatibility between their conda `pyqt`
  installation, and the new pip "`PyQt5`" installation. *This will very likely
  lead to a broken environment, forcing the user to re-create their entire
  environment and re-install napari*. This is an unfortunate consequence of
  [package naming
  decisions](https://github.com/ContinuumIO/anaconda-issues/issues/1554), and
  it's not something napari can fix.
- Alternatively, they may end up with *both* PyQt and PySide in their
  environment, and while that's not always guaranteed to break things, it can
  lead to unexpected and difficult to debug problems.

- **Don't import from PyQt5 or PySide2 in your plugin: use `qtpy`.**

    If you use `from PyQt5 import QtCore` (or similar) in your plugin, but the
    end-user has chosen to use `PySide2` for their Qt backend — or vice versa —
    then your plugin will fail to import.  Instead use `from qtpy import
    QtCore`.  `qtpy` is a [Qt compatibility
    layer](https://github.com/spyder-ide/qtpy) that will import from whatever
    backend is installed in the environment.

## Try not to depend on packages that require C compilation if these packages do not offer wheels

````{tip}
This requires some awareness of how your dependencies are built and distributed...

Some python packages write a portion of their code in lower level languages like
C or C++ and compile that code into "C Extensions" that can be called by python
at runtime.  This can *greatly* improve performance, but it means that the
package must be compiled for *each* platform (i.e. Windows, Mac, Linux) that the
package wants to support.  Some packages do this compilation step ahead of time,
by distributing "[wheels](https://realpython.com/python-wheels/)" on
[PyPI](https://pypi.org/)... or by providing pre-compiled packages via `conda`.
Other packages simply distribute the source code (as an "sdist") and expect the
end-user to compile it on their own computer.  Compiling C code requires
software that is not always installed on every computer. (If you've ever tried
to `pip install` a package and had it fail with a big wall of red text saying
something about `gcc`, then you've run into a package that doesn't distribute
wheels, and you didn't have the software required to compile it).
````


As a plugin developer, if you depend on a package that uses C extensions but
doesn't distribute a pre-compiled wheel, then it's very likely that your users
will run into difficulties installing your plugin:

- *What is a "wheel"?*

  Briefly, a wheel is a *built distribution*, containing code that is
  pre-compiled for a specific operating system.

  For more detail, see [What Are Python Wheels and Why Should You
  Care?](https://realpython.com/python-wheels/)

- *How do I know if my dependency offers a wheel*

  There are many ways, but a sure-fire way to know is to go to the respective
  package on PyPI, and click on the "Download Files" link.  If the package
  offers wheels, you'll see one or more files ending in `.whl`.  For example,
  [napari offers a wheel](https://pypi.org/project/napari/#files).  If a package
  *doesn't* offer a wheel, it may still be ok if it's just a pure python package
  that doesn't have any C extensions...

- *How do I know if one of my dependencies uses C Extensions?*

  There's no one right way, but more often than not, if a package uses C
  extensions, the `setup()` function in their `setup.py` file will use the
  [`ext_modules`
  argument](https://docs.python.org/3/distutils/setupscript.html#describing-extension-modules).
  (for example, [see here in
  pytorch](https://github.com/pytorch/pytorch/blob/master/setup.py#L914))

````{admonition} What about conda?
**conda** also distributes & installs pre-compiled packages, though they aren't
wheels.  While this is definitely a fine way to install binary dependencies in a
reliable way, the built-in napari plugin installer doesn't currently work with
conda.  If your dependency is only available on conda, but does not offer
wheels,you *may* guide your users in using conda to install your package or one
of your dependencies.  Just know that it may not work with the built-in plugin
installer. 
````


## Don't import heavy dependencies at the top of your module

````{note}
This point will be less relevant when we move to the second generation
[manifest-based plugin
declaration](https://github.com/napari/napari/issues/3115), but it's still a
good idea to delay importing your plugin-specific dependencies and modules until
*after* your hookspec has been called.  This helps napari stay quick and
responsive at startup.  
````



Consider the following example plugin:

```ini
[options.entry_points]
napari.plugin =
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

This can deterioate the end-user experience, and make napari feel slugish. Best
practice is to delay heavy imports until right before they are used.  The
following slight modification will help napari load much faster:

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

## Don't leave resources open

It's always good practice to clean up resources like open file handles and
databases.  As a napari plugin it's particularly important to do this (and
especially for Windows users).  If someone tries to use the built-in plugin
manager to *uninstall* your plugin, open file handles and resources may cause
the process to fail or even leave your plugin in an "installed-but-unuseable"
state.

Don't do this:

```py
# my_plugin/module.py
import json

data_file = open("some_data_in_my_plugin.json")
data = json.load(data_file)
```

Instead, make sure to close your resource after grabbing the data (ideally by
using a context manager, but manually otherwise):

```py
with open("some_data_in_my_plugin.json") as data_file:
    data = json.load(data_file)
```



## Write extensive tests for your plugin!

Programmer and author Bruce Eckel famously wrote:

> "If it's not tested, it's broken"

It's true.  High test coverage is one way to show your users that you are
dedicated to the stability of your plugin. Aim for 100%!

Of course, simply having 100% coverage doesn't mean your code is bug-free, so
make sure that you test all of the various ways that your code might be called.

See our [Tips for testing napari plugins](plugin-testing-tips).

### How to check test coverage?

The [cookiecutter
template](https://github.com/napari/cookiecutter-napari-plugin) is already set
up to report test coverage, but you can test locally as well, using
[pytest-cov](https://github.com/pytest-dev/pytest-cov)

1. `pip install pytest-cov`
2. Run your tests with `pytest --cov=<your_package> --cov-report=html`
3. Open the resulting report in your browser: `open htmlcov/index.html`
4. The report will show line-by-line what is being tested, and what is being
   missed. Continue writing tests until everything is covered! If you have
   lines that you *know* never need to be tested (like debugging code) you can
   [exempt specific
   lines](https://coverage.readthedocs.io/en/6.4.4/excluding.html#excluding-code-from-coverage-py)
   from coverage with the comment `# pragma: no cover`
5. In the cookiecutter, coverage tests from github actions will be uploaded to
   codecov.io

## Set style for additional windows in your plugin

In napari plugins we strongly advise additional widgets be docked in the main napari viewer,
but sometimes a separate window is required. 
The best practice is to use [`QDialog`](https://doc.qt.io/qt-5/qdialog.html)
based windows with parent set to widget
already docked in the viewer.

```python
from qtpy.QtWidgets import QDialog, QWidget, QSpinBox, QPushButton, QGridLayout, QLabel

class MyInputDialog(QDialog):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setWindowTitle("My Input Dialog")
        self.number = QSpinBox()
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")
        
        layout = QGridLayout()
        layout.addWidget(QLabel("Number:"), 0, 0)
        layout.addWidget(self.number, 0, 1)
        layout.addWidget(self.ok_btn, 1, 0)
        layout.addWidget(self.cancel_btn, 1, 1)
        self.setLayout(layout)
        
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
class MyWidget(QWidget):
    def __init__(self, viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.open_dialog = QPushButton("Open dialog")
        self.open_dialog.clicked.connect(self.open_dialog_clicked)
        
    def open_dialog_clicked(self):
        # setting parent to self allows the dialog to inherit its 
        # style from the viewer by pass self as argument
        dialog = MyInputDialog(self)  
        dialog.exec_()
        if dialog.result() == QDialog.Accepted:
            print(dialog.number.value())
```

If there is a particular reason that you need to use a separate window that
inherits from `QWidget`, not `QDialog`, then you could use the `get_current_stylesheet` 
and [`get_stylesheet`](/api/napari.qt.html#napari.qt.get_stylesheet) functions from the [`napari.qt`](/api/napari.qt.html) module.

Here is a `magicgui` example (but could be easily generalised to native `qt` based widgets):

```python
from magicgui import magicgui

from napari.qt import get_current_stylesheet
from napari.settings import get_settings

def sample_add(a: int, b: int) -> int:
    return a + b

@magicgui
def sample_add(a: int, b: int) -> int:
    return a + b

def change_style():
    sample_add.native.setStyleSheet(get_current_stylesheet())


get_settings().appearance.events.theme.connect(change_style)
change_style()

```