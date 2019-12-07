from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QStackedWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QDialog,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QComboBox,
    QLabel,
)
from pathlib import Path
import os.path
import napari
from .layers.qt_image_layer import QtImageDialog


class QtOpenDialog(QDialog):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer
        self._add_methods = {napari.layers.Image: self.viewer.add_image}
        self.widgets = {'Image': QtImageDialog()}

        self._last_visited_dir = str(Path.home())

        self.layout = QVBoxLayout()

        # Name of file to be loaded
        self.fileName = QLineEdit()
        self.fileName.setToolTip('File name')
        self.fileName.setAcceptDrops(False)
        self.layout.addWidget(self.fileName)

        self.fileList = QListWidget()
        self.fileList.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.fileList.setAutoScroll(False)
        self.layout.addWidget(self.fileList)

        self.selection = QWidget()
        self.selectionLayout = QHBoxLayout()
        self.selection.setLayout(self.selectionLayout)

        # remove files
        self.removeFiles = QPushButton('remove files')
        self.removeFiles.setObjectName('dialog')
        self.removeFiles.clicked.connect(self._click_remove_files)
        self.selectionLayout.addWidget(self.removeFiles)

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

        # layer type selection
        self.layerTypeComboBox = QComboBox()
        for name in self.widgets.keys():
            self.layerTypeComboBox.addItem(name)
        self.layerTypeComboBox.activated[str].connect(
            lambda text=self.layerTypeComboBox: self.change_layer_type(text)
        )
        self.layerTypeComboBox.setCurrentText(
            str(napari.layers.Image.__name__)
        )
        layer_type_layout = QHBoxLayout()
        layer_type_layout.addWidget(QLabel('Layer type:'))
        layer_type_layout.addWidget(self.layerTypeComboBox)
        layer_type_layout.setSpacing(0)
        self.layout.addLayout(layer_type_layout)

        self.stack = QStackedWidget()
        for widget in self.widgets.values():
            self.stack.addWidget(widget)
        self.layout.addWidget(self.stack)

        self.fileOpen = QWidget()
        self.fileOpenLayout = QHBoxLayout()
        self.fileOpenLayout.addStretch(1)

        # cancel
        self.cancel = QPushButton('cancel')
        self.cancel.setObjectName('dialog')
        self.cancel.clicked.connect(self.close)
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
        self.setAcceptDrops(True)
        self.setFocus()

    @staticmethod
    def show_dialog(qt_viewer):
        d = QtOpenDialog(qt_viewer.viewer)
        d.setStyleSheet(qt_viewer.styleSheet())
        d.setWindowTitle('Add layer')
        d.setWindowModality(Qt.ApplicationModal)
        d.exec_()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return and self.fileName.hasFocus():
            self._finish_text()
            event.accept()
        elif (
            event.key() == Qt.Key_Return
            and self.open.isEnabled()
            and self.hasFocus()
        ):
            event.accept()
            self._finish_dialog()
        elif event.key() == Qt.Key_Return and self.hasFocus():
            event.accept()
            self._click_select_files()
        elif event.key() == Qt.Key_Escape:
            self.setFocus()
            self._refresh_focus()
        elif event.key() == Qt.Key_Backspace:
            self._click_remove_files()

    def change_layer_type(self, text):
        self.stack.setCurrentWidget(self.widgets[text])

    def _finish_text(self):
        """Add new file from text edit box."""
        text = self.fileName.text()
        if len(text) > 0:
            item = QListWidgetItem()
            item.setText(text)
            self.fileList.addItem(item)
            self.fileName.setText("")
        else:
            self.fileName.clearFocus()
            self.setFocus()
        self._refresh_focus()

    def _click_remove_files(self):
        """Remove selected files from list."""
        selected_items = self.fileList.selectedItems()
        for item in selected_items:
            self.fileList.takeItem(self.fileList.row(item))
            del item
        self._refresh_focus()

    def _refresh_focus(self):
        enabled = self.fileList.count() > 0
        self.open.setEnabled(enabled)
        self.open.setProperty('highlight', enabled)
        self.open.style().polish(self.open)
        self.selectFiles.setProperty('highlight', not enabled)
        self.selectFiles.style().polish(self.selectFiles)

    def get_filenames(self):
        """Get non-empty filenames from textbox.

        Returns
        ----------
        filenames : list of str
            List of filenames
        """
        filenames = []
        for index in range(self.fileList.count()):
            filenames.append(self.fileList.item(index).text())
        return filenames

    def _click_select_files(self):
        """Open file using native file open dialog."""
        filenames, _ = QFileDialog.getOpenFileNames(
            parent=self,
            caption='Select file(s)',
            directory=self._last_visited_dir,  # home dir by default
        )
        if filenames is not None:
            for text in filenames:
                item = QListWidgetItem()
                item.setText(text)
                self.fileList.addItem(item)
        self.fileName.setText("")
        self._refresh_focus()

    def _click_select_folder(self):
        """Open folder using native directory open dialog."""
        folder = QFileDialog.getExistingDirectory(
            parent=self,
            caption='Select folder',
            directory=self._last_visited_dir,  # home dir by default
        )
        if folder is not None:
            item = QListWidgetItem()
            item.setText(folder)
            self.fileList.addItem(item)
        self.fileName.setText("")
        self._refresh_focus()

    def _finish_dialog(self):
        if self.fileList.count() > 0:
            filenames = self.get_filenames()
            self._last_visited_dir = os.path.dirname(filenames[0])
            widget = self.stack.currentWidget()
            arguments = widget.get_arguments()
            add_method = self._add_methods[widget.layer]
            add_method(path=filenames, **arguments)
        self.close()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Add local files and web URLS with drag and drop."""
        filenames = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                filenames.append(url.toLocalFile())
            else:
                filenames.append(url.toString())
        if len(filenames) > 0:
            for text in filenames:
                item = QListWidgetItem()
                item.setText(text)
                self.fileList.addItem(item)
        self.fileName.setText("")
        self._refresh_focus()
