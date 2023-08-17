from typing import List, Tuple

from qtpy.QtWidgets import QFileDialog, QWidget

from napari.utils.misc import in_ipython


class class_or_instancemethod(classmethod):
    def __get__(self, instance, type_):
        descr_get = (
            super().__get__ if instance is None else self.__func__.__get__
        )
        return descr_get(instance, type_)


class QCustomFileDialog(QFileDialog):
    def _setup_dialog(self, parent, directory, options):
        if parent is not None:
            self.setParent(parent)
        self.setDirectory(directory)
        if in_ipython():
            options = options | QFileDialog.Option.DontUseNativeDialog
        self.setOption(options)

    @class_or_instancemethod
    def getExistingDirectory(
        self_or_cls,
        parent: QWidget = None,
        caption: str = '',
        directory: str = '',
        options=QFileDialog.Option.ShowDirsOnly,
    ):
        obj = self_or_cls() if isinstance(self_or_cls, type) else self_or_cls
        obj._setup_dialog(parent, directory, options)

        obj.setFileMode(QFileDialog.FileMode.Directory)
        obj.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)

        if obj.exec_():
            return obj.selectedFiles()[0]
        return None

    @class_or_instancemethod
    def getOpenFileNames(
        self_or_cls,
        parent: QWidget = None,
        caption: str = '',
        directory: str = '',
        filter: str = '',  #  noqa: A002
        options=QFileDialog.Option(0),  # noqa: B008
    ) -> Tuple[List[str], str]:
        obj = self_or_cls() if isinstance(self_or_cls, type) else self_or_cls
        obj._setup_dialog(parent, directory, options)

        obj.setFileMode(QFileDialog.FileMode.ExistingFiles)
        obj.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        if filter:
            obj.setNameFilter(filter)

        if obj.exec_():
            return obj.selectedFiles(), obj.selectedNameFilter()
        return [], ''

    @class_or_instancemethod
    def getSaveFileName(
        self_or_cls,
        parent: QWidget = None,
        caption: str = '',
        directory: str = '',
        filter: str = "",  # noqa: A002
        options=QFileDialog.Option(0),  # noqa: B008
    ) -> Tuple[str, str]:
        obj = self_or_cls() if isinstance(self_or_cls, type) else self_or_cls
        obj._setup_dialog(parent, directory, options)

        obj.setFileMode(QFileDialog.FileMode.AnyFile)
        obj.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        if filter:
            obj.setNameFilter(filter)

        if obj.exec_():
            return obj.selectedFiles()[0], obj.selectedNameFilter()
        return '', ''
