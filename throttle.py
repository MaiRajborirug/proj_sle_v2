"""In-memory abuse throttle for the public submission endpoint.

The app is a public URL accepting anonymous submissions, so nothing stops a bot from
flooding the sheet — and because submissions are anonymous, junk rows cannot be
identified and removed afterwards. This caps how fast any one client can submit.

Client IPs are hashed with a per-deployment salt and held **in memory only**. Nothing here
is ever written to the sheet or to disk, so no personal data is stored. Mention this
transient hashing in the ethics application anyway.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from collections import defaultdict, deque

MAX_EVENTS = 10
WINDOW_SECONDS = 600
_STALE_AFTER = WINDOW_SECONDS * 2

# A random salt is generated if none is configured. That is fine — the hash only needs to
# be stable for the lifetime of the process, and a fresh salt per restart is strictly more
# private than a fixed one.
_SALT = os.environ.get("IP_HASH_SALT") or secrets.token_hex(16)


class Throttle:
    """Sliding-window rate limiter keyed on a salted hash of the client IP."""

    def __init__(self, max_events: int = MAX_EVENTS, window_seconds: int = WINDOW_SECONDS):
        """Initialise an empty throttle.

        Args:
            max_events: Submissions allowed per client within the window.
            window_seconds: Length of the sliding window, in seconds.
        """
        self._max = max_events
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._last_swept = 0.0

    def _key(self, client_ip: str) -> str:
        return hashlib.sha256(f"{_SALT}{client_ip}".encode()).hexdigest()

    def _sweep(self, now: float) -> None:
        """Drop clients that have been silent for long enough to be irrelevant."""
        if now - self._last_swept < _STALE_AFTER:
            return
        self._last_swept = now
        stale = [k for k, hits in self._hits.items() if not hits or now - hits[-1] > _STALE_AFTER]
        for k in stale:
            del self._hits[k]

    def allow(self, client_ip: str) -> bool:
        """Record an attempt and report whether it is within the limit.

        Args:
            client_ip: The client address. An empty string disables throttling for this
                call, which is what happens when the host does not expose a client IP.

        Returns:
            True if the submission should proceed, False if the client is over its limit.
        """
        if not client_ip:
            return True
        now = time.monotonic()
        self._sweep(now)
        hits = self._hits[self._key(client_ip)]
        while hits and now - hits[0] > self._window:
            hits.popleft()
        if len(hits) >= self._max:
            return False
        hits.append(now)
        return True
