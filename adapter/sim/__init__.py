"""Simulator and dataset adapters for embodied.cpp."""

__all__ = [
    "Evo1LIBEROParser",
    "GR00TN16LIBEROParser",
    "LIBEROSimAdapter",
    "LeRobotLIBEROParser",
    "RobotWinHyVLAAdapter",
    "RobotWinObservationAdapter",
]


def __getattr__(name):
    if name in {"RobotWinHyVLAAdapter", "RobotWinObservationAdapter"}:
        from . import robotwin
        return getattr(robotwin, name)
    if name in {
        "Evo1LIBEROParser",
        "GR00TN16LIBEROParser",
        "LIBEROSimAdapter",
        "LeRobotLIBEROParser",
    }:
        from . import libero
        return getattr(libero, name)
    raise AttributeError(name)
