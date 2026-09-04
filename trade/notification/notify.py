"""Notification interface used by live-trading orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Notify(ABC):
    """Deliver operational notifications without exposing a provider API."""

    @abstractmethod
    def send(self, message: str) -> bool:
        """Send one notification and return whether delivery succeeded."""

        raise NotImplementedError
