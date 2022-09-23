# napari + ImageJ how-to guide

ImageJ is a Java-based image processing program that provides extensibility via Java plugins and recordable macros. It can display, edit, analyze, process, save, and print 8-bit color and grayscale, 16-bit integer, and 32-bit floating point images. It can read many image file formats, including TIFF, PNG, GIF, JPEG, BMP, DICOM, and FITS, as well as raw formats. It has a plethora of features that can be checked out [here](https://en.wikipedia.org/wiki/ImageJ#Features).

People who wish to try their hands on ImageJ can do so by downloading and installing it using this [link](https://imagej.net/software/fiji/downloads)

## Reading images with ImageJ and viewing them with napari

Here we first cut at reading images with SCIFIO+Bio-Formats via PyimageJ into NumPy arrays
and then display them with Napari.

```python
import napari, sys

if len(sys.argv) <= 1:
    print('Please specify one or more images as arguments.')
    exit(1)

try:
    import imagej
except ModuleNotFoundError:
    raise ModuleNotFoundError("""This example uses ImageJ but pyimagej is not
    installed. To install try 'conda install pyimagej'.""")

print('--> Initializing imagej')
ij = imagej.init('sc.fiji:fiji') # Fiji includes Bio-Formats.

viewer = napari.Viewer()
for path in sys.argv[1:]:
    print('--> Reading {}'.format(path))

    dataset = ij.io().open(path)
    image = ij.py.from_java(dataset)

    viewer.add_image(image)

ij.getContext().dispose()

viewer.grid_view()
napari.run()
```

## Using ImageJ and napari side-by-side

### Issues with using ImageJ and napari simultaneously

- Threading concerns with macOS i.e. to display the Fiji user interface, Java AWT must be started on the Cocoa event loop, however, attempting to invoke the napari viewer from the Cocoa event loop thread crashes the program with errors. 
- If we code a script to open napari and ImageJ together then
  - The napari UI starts,
  - But, the script blocks until the close of napari viewer window.
  - Even after closing the window, the ImageJ UI never appears even though the code to start ImageJ does then execute.
 
### Simultaneously using ImageJ and napari in various environments

Due to behavioural differences between plain Python and IPython we use slightly different approaches to run ImageJ and python simultaneously in each different environment. 

#### 1. Running napari+ImageJ from plain Python

Firstly import napari:

```python
import napari
viewer = napari.Viewer()
```

When napari comes up, open the Jupyter Qt console and type:

```python
import imagej
ij = imagej.init(headless=False)
ij.ui().showUI()
```

This works because the console in napari is running in the correctly initialized Qt GUI/main thread. However,if we even touch the Java UI from Python it locks up i.e.  Python will now lock up even for the simplest line of code such as:

```python 
ij.ui().showDialog('hello')
```

To fix this we can use either of these 3 following methods :

1. Use Java’s EventQueue to queue the task on the Java event dispatch thread:
    ```python 
    from jnius import PythonJavaClass, java_method, autoclass
    class JavaRunnable(PythonJavaClass):
        __javainterfaces__ = ['java/lang/Runnable']
        def __init__(self, f):
            super(JavaRunnable, self).__init__()
            self._f = f
    
        @java_method('()V')
        def run(self):
            self._f()
    
    EventQueue = autoclass('java.awt.EventQueue')
    EventQueue.invokeLater(JavaRunnable(lambda: ij.ui().showDialog('hello')))
    ```

2. Use the SciJava ScriptService:
    ```python
    ij.script().run('.groovy', "#@ ImageJ ij\nij.ui().showDialog('hello')", True)
    ```

3. Using ImageJ’s [Script Editor](https://imagej.net/scripting/script-editor)

#### 2. Starting napari + ImageJ from plain Python (without napari's Qt Console)
Here is a plain Python script that starts up Qt and spins up ImageJ without use of napari's Qt Console. 

``` python
from PyQt5 import QtCore, QtWidgets

def main():
    app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed( True )

    def start_imagej():
        import imagej
        ij = imagej.init(headless=False)
        print(ij.getVersion())
        ij.launch()
        print("Launched ImageJ")

    QtCore.QTimer.singleShot(0, start_imagej)
    app.exec_()

if __name__ == "__main__":
    main()
```

Note that the app.exec_() call blocks the main thread, because Qt takes it over as its GUI/main thread. On macOS, the main thread is the only thread that works for Qt to use as its GUI/main thread.

#### 3. Starting napari+ImageJ from IPython

A code that successfully starts ImageJ from IPython 

**NOTE:** First initialize Qt using %gui qt or at launch via ipython --gui=qt

```python
def start_imagej():
    import imagej
    global ij
    ij = imagej.init(headless=False)
    ij.ui().showUI()
    print(ij.getVersion())
from PyQt5 import QtCore
QtCore.QTimer.singleShot(0, start_imagej)
```

This how-to guide is an adaptation of demos provided by [Curtis Rueden](https://forum.image.sc/u/ctrueden) on [https://forum.image.sc/](https://forum.image.sc/t/read-images-with-imagej-display-them-with-napari/32156) platform.
