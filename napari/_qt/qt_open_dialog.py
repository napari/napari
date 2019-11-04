from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QDialog,
    QPlainTextEdit,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QLineEdit,
)
from pathlib import Path
import os.path
import napari
from ..util.misc import get_keybindings_summary
from .layers.qt_image_layer import QtImageDialog


class QtOpenDialog(QWidget):
    def __init__(self, viewer, parent):
        super(QtOpenDialog, self).__init__(parent)

        self.viewer = viewer
        self._last_visited_dir = str(Path.home())

        self.layout = QVBoxLayout()

        # Name of file to be loaded
        self.fileName = QLineEdit()
        # textbox.setText(layer.name)
        # textbox.home(False)
        # textbox.setToolTip('Layer name')
        # textbox.setAcceptDrops(False)
        # textbox.setEnabled(True)
        # textbox.editingFinished.connect(self.changeText)
        self.layout.addWidget(self.fileName)

        self.selection = QWidget()
        self.selectionLayout = QHBoxLayout()

        # select folder
        self.selectFolder = QPushButton('select folder')
        self.selectFolder.setObjectName('dialog')
        self.selectFolder.clicked.connect(self._click_select_folder)
        self.selectionLayout.addWidget(self.selectFolder)

        # select files
        self.selectFiles = QPushButton('select file(s)')
        self.selectFiles.setObjectName('dialog')
        self.selectFiles.setProperty('highlight', True)
        self.selectFiles.clicked.connect(self._click_select_files)
        self.selectionLayout.addWidget(self.selectFiles)

        self.layout.addWidget(self.selection)

        self.tabs = QTabWidget()
        self.tabs.addTab(QtImageDialog(napari.layers.Image), 'Image')
        self.tabs.addTab(QtLayerOpen(napari.layers.Labels), 'Labels')
        self.tabs.addTab(QtLayerOpen(napari.layers.Points), 'Labels')
        self.tabs.addTab(QtLayerOpen(napari.layers.Shapes), 'Shapes')
        self.tabs.addTab(QtLayerOpen(napari.layers.Surface), 'Surface')
        self.tabs.addTab(QtLayerOpen(napari.layers.Vectors), 'Vectors')
        self.layout.addWidget(self.tabs)

        # # file paths
        # self.pathTextBox = QPlainTextEdit(self)
        # self.pathTextBox.setPlainText("")
        # self.pathTextBox.setToolTip('File paths')
        # self.pathTextBox.setAcceptDrops(False)
        # self.pathTextBox.setEnabled(True)
        # self._default_cursor_width = self.pathTextBox.cursorWidth()
        # self.pathTextBox.clearFocus()
        # self.pathTextBox.textChanged.connect(self._on_text_change)
        # self.pathTextBox.focusOutEvent = self._focus_text_out
        # self.pathTextBox.focusInEvent = self._focus_text_in
        # self.layout.addWidget(self.pathTextBox)

        self.fileOpen = QWidget()
        self.fileOpenLayout = QHBoxLayout()
        self.fileOpenLayout.addStretch(1)

        # cancel
        self.cancel = QPushButton('cancel')
        self.cancel.setObjectName('dialog')
        self.cancel.clicked.connect(parent.close)
        self.fileOpenLayout.addWidget(self.cancel)

        # open
        self.open = QPushButton('open')
        self.open.setObjectName('open')
        self.open.setEnabled(False)
        self.open.setProperty('highlight', False)
        self.open.clicked.connect(self._finish_dialog)

        self.fileOpenLayout.addWidget(self.open)
        self.fileOpen.setLayout(self.fileOpenLayout)
        self.layout.addWidget(self.fileOpen)

        self.setLayout(self.layout)
        self.setFocus()

    @staticmethod
    def showDialog(qt_viewer):
        d = QDialog()
        d.setObjectName('OpenDialog')
        d.setStyleSheet(qt_viewer.styleSheet())
        d.setWindowTitle('add layer')
        qt_viewer._open = QtOpenDialog(qt_viewer.viewer, d)
        d.setWindowModality(Qt.ApplicationModal)
        d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        qt_viewer._open_dialog = d
        d.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return and self.open.isEnabled():
            event.accept()
            self._finish_dialog()
        elif event.key() == Qt.Key_Return:
            event.accept()
            self._click_select_files()
        elif event.key() == Qt.Key_Escape:
            self.pathTextBox.clearFocus()
            self.setFocus()

    def _on_text_change(self):
        """If filenames are present enable open button."""
        filenames = self.get_filenames()
        self.open.setEnabled(len(filenames) > 0)

    def _focus_text_out(self, event):
        if event is not None:
            QPlainTextEdit.focusOutEvent(self.pathTextBox, event)
        self.pathTextBox.setCursorWidth(0)
        filenames = self.get_filenames()
        enabled = len(filenames) > 0
        self.open.setEnabled(enabled)
        self.open.setProperty('highlight', enabled)
        self.open.style().polish(self.open)
        self.selectFiles.setProperty('highlight', not enabled)
        self.selectFiles.style().polish(self.selectFiles)

    def _focus_text_in(self, event):
        if event is not None:
            QPlainTextEdit.focusInEvent(self.pathTextBox, event)
        self.pathTextBox.setCursorWidth(self._default_cursor_width)
        self.open.setProperty('highlight', False)
        self.open.style().polish(self.open)
        self.selectFiles.setProperty('highlight', False)
        self.selectFiles.style().polish(self.selectFiles)

    def get_filenames(self):
        """Get non-empty filenames from textbox.

        Returns
        ----------
        filenames : list of str
            List of filenames
        """
        raw_filenames = self.pathTextBox.toPlainText()
        filenames = raw_filenames.splitlines()
        return [f for f in filenames if len(f) > 0]

    def _click_select_files(self):
        """Open file using native file open dialog."""
        filenames, _ = QFileDialog.getOpenFileNames(
            parent=self,
            caption='Select file(s)',
            directory=self._last_visited_dir,  # home dir by default
        )
        if filenames is None:
            self.pathTextBox.setPlainText("")
        else:
            self.pathTextBox.setPlainText("\n".join(filenames))
        if self.pathTextBox.hasFocus():
            self.pathTextBox.clearFocus()
            self.setFocus()
        else:
            self._focus_text_out(None)

    def _click_select_folder(self):
        """Open folder using native directory open dialog."""
        folder = QFileDialog.getExistingDirectory(
            parent=self,
            caption='Select folder',
            directory=self._last_visited_dir,  # home dir by default
        )
        if folder is None:
            self.pathTextBox.setPlainText("")
        else:
            self.pathTextBox.setPlainText(folder)
        if self.pathTextBox.hasFocus():
            self.pathTextBox.clearFocus()
            self.setFocus()
        else:
            self._focus_text_out(None)

    def _finish_dialog(self):
        filenames = self.get_filenames()
        if len(filenames) > 0:
            self._last_visited_dir = os.path.dirname(filenames[0])
            arguments = self.tabs.currentWidget().get_arguments()
            self.viewer.add_image(path=filenames, **arguments)
        self.parent().close()


# class QtFileNamesList(QScrollArea):
#     def __init__(self):
#         super().__init__()
#
#         self.setWidgetResizable(True)
#         self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
#         scrollWidget = QWidget()
#         self.setWidget(scrollWidget)
#         self.vbox_layout = QVBoxLayout(scrollWidget)
#         self.vbox_layout.addWidget(QLineEdit())
#         self.vbox_layout.addStretch(1)
#         self.vbox_layout.setContentsMargins(0, 0, 0, 0)
#         self.vbox_layout.setSpacing(2)
#
# class QtFileName(QWidget):
#     def __init__(self):
#         super().__init__()


class QtLayerOpen(QWidget):
    def __init__(self, layer):
        super().__init__()

        self.layout = QVBoxLayout()

        # # Add class keybindings for the layer
        # if len(layer.class_keymap) == 0:
        #     keybindings_str = 'No keybindings'
        # else:
        #     keybindings_str = get_keybindings_summary(layer.class_keymap)
        #
        # layer_label = QLabel(keybindings_str)
        # layer_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # # layer_label.setAlignment(Qt.AlignLeft)
        # layer_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.layout.addWidget(layer_label)
        self.setLayout(self.layout)
