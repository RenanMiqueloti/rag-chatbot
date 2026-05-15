"""Helpers de rate limiting reutilizados entre api.py e gradio_app.py.

Duas camadas independentes:
  * ``is_rate_limit`` — detecta erro de upstream (Groq/OpenAI/Anthropic) pra
    transformar a exceção num 429 amigável.
  * ``DailyRequestBudget`` — circuit breaker in-memory com reset em meia-noite
    UTC. Protege a quota diária do provider quando muitos IPs distintos
    consomem em paralelo (slowapi por IP não cobre esse caso).
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

RATE_LIMIT_MSG = "Limite de requisições do provider atingido. Tente em ~1 minuto."
DAILY_CAP_MSG = "Cota diária da demo atingida. Volte amanhã."


def is_rate_limit(exc: BaseException) -> bool:
    """Heurística: True se a exceção parece um 429/rate limit do upstream LLM."""
    s = f"{type(exc).__name__} {exc!s}".lower()
    return "ratelimit" in s or "rate_limit" in s or "429" in s


class DailyRequestBudget:
    """Counter in-memory com reset automático em meia-noite UTC.

    ``cap <= 0`` desabilita o budget (``try_consume`` sempre True).
    Thread-safe via lock — uvicorn com 1 worker é o caso alvo; multi-worker
    perderia consistência (cada worker tem seu próprio counter).
    """

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self._used = 0
        self._reset_at = self._next_midnight_utc()
        self._lock = threading.Lock()

    @staticmethod
    def _next_midnight_utc() -> datetime:
        now = datetime.now(UTC)
        return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    def _maybe_reset_locked(self) -> None:
        if datetime.now(UTC) >= self._reset_at:
            self._used = 0
            self._reset_at = self._next_midnight_utc()

    def remaining(self) -> int:
        """Retorna requests restantes hoje, ou ``-1`` se o budget está desabilitado."""
        if self.cap <= 0:
            return -1
        with self._lock:
            self._maybe_reset_locked()
            return max(0, self.cap - self._used)

    def try_consume(self, amount: int = 1) -> bool:
        """True se conseguiu consumir; False se estouraria o cap."""
        if self.cap <= 0:
            return True
        with self._lock:
            self._maybe_reset_locked()
            if self._used + amount > self.cap:
                return False
            self._used += amount
            return True
