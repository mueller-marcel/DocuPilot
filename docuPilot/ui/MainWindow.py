from PySide6.QtWidgets import QMainWindow

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        """
        Initialize the main window
        """
        
        super().__init__()
        self.setWindowTitle("DocuPilot")
        self.resize(800, 500)