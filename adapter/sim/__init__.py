"""Simulator and dataset adapters for embodied.cpp."""

__all__ = [
    "LIBEROSimAdapter",
    "LingBotLIBEROParser",
    "RobotWinHyVLAAdapter",
    "RobotWinObservationAdapter",
]


def __getattr__(name):
    if name in {"RobotWinHyVLAAdapter", "RobotWinObservationAdapter"}:
        from . import robotwin
        return getattr(robotwin, name)
    if name in {
        "LIBEROSimAdapter",
        "LingBotLIBEROParser",
    }:
        from . import libero
        return getattr(libero, name)
    raise AttributeError(name)
