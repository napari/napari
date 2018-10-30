import sys
from PyQt5.QtWidgets import QApplication, QAction


class NapariApplication(QApplication):
    """Base Napari application"""
    def __init__(self, List):
        super(QApplication, self).__init__(List)


def main():
    application = NapariApplication(sys.argv)

    # do stuff here

    sys.exit(application.exec_())
