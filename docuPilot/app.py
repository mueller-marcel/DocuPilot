from PySide6.QtWidgets import QApplication
import sys

from docuPilot.ui.MainWindow import MainWindow


def run():
    """
    Bootstrap the application and run it
    """

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()