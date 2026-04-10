from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget

from docupilot.ui.widgets.ScreenSelectorWidget import ScreenSelectorWidget
from docupilot.ui.widgets.MicrophoneSelectorWidget import MicrophoneSelectorWidget
from docupilot.ui.widgets.RecordButtonWidget import RecordButtonWidget

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        """
        Initializes the main window.
        """

        super().__init__()
        self.setWindowTitle("DocuPilot")
        self.resize(1000, 750)

        self.selected_screen = None
        self.selected_microphone = None
        self.record_button_widget = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """
        Set up the user interface.
        """

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        box_layout = QVBoxLayout(central_widget)

        self.screen_selector = ScreenSelectorWidget()
        self.screen_selector.screen_selected.connect(self.on_screen_selected)

        self.microphone_selector = MicrophoneSelectorWidget()
        self.microphone_selector.microphone_selected.connect(self.on_microphone_selected)

        self.record_button_widget = RecordButtonWidget()
        self.record_button_widget.record_started.connect(self.on_record_started)
        self.record_button_widget.record_stopped.connect(self.on_record_stopped)

        box_layout.addWidget(self.screen_selector)
        box_layout.addWidget(self.microphone_selector)
        box_layout.addStretch()
        box_layout.addWidget(self.record_button_widget)

    def on_screen_selected(self, screen) -> None:
        """
        Handles the screen-selected event.
        :param screen: The screen that has been selected.
        """

        self.selected_screen = screen

    def on_microphone_selected(self, microphone) -> None:
        """
        Handles the microphone-selected event.
        :param microphone: The microphone that has been selected.
        """

        self.selected_microphone = microphone

    @staticmethod
    def on_record_started() -> None:
        """
        Triggered when the recording has started.
        :return: None
        """

        print("Aufzeichnung gestartet")

    @staticmethod
    def on_record_stopped() -> None:
        """
        Triggered when the recording has stopped.
        :return: None
        """

        print("Aufzeichnung gestoppt")