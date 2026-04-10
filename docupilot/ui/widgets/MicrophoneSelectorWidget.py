from PySide6.QtCore import Signal
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtWidgets import QWidget

from docupilot.ui.widgets.DeviceSelectorWidget import DeviceSelectorWidget
from docupilot.ui.widgets.MicrophoneTileWidget import MicrophoneTileWidget


class MicrophoneSelectorWidget(DeviceSelectorWidget):
    microphone_selected = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Initialize the microphone selector widget.
        :param parent: The reference to the parent widget.
        """

        super().__init__(
            title="Mikrofon auswählen",
            load_devices=lambda: QMediaDevices.audioInputs(),
            tile_factory=lambda microphone: MicrophoneTileWidget(microphone),
            selected_attr_name="microphone",
            default_device_resolver=lambda devices: QMediaDevices.defaultAudioInput(),
            columns=3,
            parent=parent,
        )

        self.device_selected.connect(self.microphone_selected.emit)

    def reload_microphones(self) -> None:
        """
        Reload the available microphones.
        """

        self.reload_devices()

    def get_selected_microphone(self):
        """
        Get the selected microphone and return it.
        """

        return self.get_selected_device()