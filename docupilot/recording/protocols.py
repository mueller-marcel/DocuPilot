from __future__ import annotations

from typing import Protocol


class ScreenGeometry(Protocol):
    """
    Represents the geometry of a screen (position and dimensions).
    """

    def x(self) -> int:
        ...

    def y(self) -> int:
        ...

    def width(self) -> int:
        ...

    def height(self) -> int:
        ...


class Screen(Protocol):
    """
    Minimal interface for a screen device.
    """

    def name(self) -> str:
        ...

    def geometry(self) -> ScreenGeometry:
        ...

    def devicePixelRatio(self) -> float:
        ...


class Microphone(Protocol):
    """
    Minimal interface for an audio input device.
    """

    def description(self) -> str:
        ...


class ModalityRecorder(Protocol):
    """
    Common interface for a single recording modality.
    """

    def start(self) -> None:
        """
        Start capturing for this modality.
        """
        ...

    def stop(self) -> None:
        """
        Stop capturing and flush any pending data.
        """
        ...