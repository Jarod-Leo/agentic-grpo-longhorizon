"""Shared, bounded MiMo client; each trajectory keeps its own message history."""

from __future__ import annotations

import json
import os
import random
import threading
import time
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path


class UserSimulatorAPIError(RuntimeError):
    """Infrastructure errors must not become task rewards."""


class MimoResponseError(UserSimulatorAPIError):
    """A safe response-validation reason, without provider text or credentials."""

    def __init__(self, reason: str, retryable: bool = True):
        super().__init__(reason)
        self.reason = reason
        self.retryable = retryable


class RateLimiter:
    def __init__(self, rpm=100, tpm=10_000_000, window=60.0):
        self.rpm, self.tpm, self.window = rpm, tpm, window
        self.interval = window / rpm
        self.starts = deque()
        self.tokens = []
        self.next_start = 0.0
        self.cooldown = 0.0
        self.lock = threading.Lock()

    def acquire(self, reservation):
        if reservation > self.tpm:
            raise UserSimulatorAPIError("Request exceeds the configured token budget")
        began = time.monotonic()
        while True:
            with self.lock:
                now = time.monotonic()
                while self.starts and self.starts[0] <= now - self.window:
                    self.starts.popleft()
                self.tokens = [
                    r
                    for r in self.tokens
                    if r["end"] is None or r["end"] > now - self.window
                ]
                delay = max(0.0, self.next_start - now, self.cooldown - now)
                if len(self.starts) >= self.rpm:
                    delay = max(delay, self.starts[0] + self.window - now)
                if sum(r["tokens"] for r in self.tokens) + reservation > self.tpm:
                    expiry = [
                        r["end"] + self.window
                        for r in self.tokens
                        if r["end"] is not None
                    ]
                    delay = max(delay, min(expiry) - now if expiry else 0.05)
                if delay <= 0:
                    ticket = {"tokens": reservation, "end": None}
                    self.tokens.append(ticket)
                    self.starts.append(now)
                    self.next_start = now + self.interval
                    return ticket, now - began
            time.sleep(max(delay, 0.001))

    def finish(self, ticket, actual=None):
        with self.lock:
            # Unknown usage (e.g. a timeout) retains the conservative reservation.
            if actual is not None:
                ticket["tokens"] = actual
            ticket["end"] = time.monotonic()

    def backoff(self, seconds):
        with self.lock:
            self.cooldown = max(self.cooldown, time.monotonic() + seconds)


class MimoClient:
    def __init__(self, config, sdk=None):
        self.model = config.get("user_model", "mimo-v2.5")
        self.max_tokens = int(config.get("max_completion_tokens", 1024))
        self.limiter = RateLimiter(
            config.get("rpm", 100), config.get("tpm", 10_000_000)
        )
        self.slots = threading.BoundedSemaphore(config.get("max_inflight", 32))
        self.log_lock = threading.Lock()
        self.path = (
            Path(os.environ["MIMO_METRICS_PATH"])
            if os.environ.get("MIMO_METRICS_PATH")
            else None
        )
        self.requests = 0
        self.wait_seconds = 0.0
        self.fatal = threading.Event()
        if sdk is None:
            from openai import OpenAI

            key = os.environ.get("MIMO_API_KEY")
            if not key:
                raise UserSimulatorAPIError("MIMO_API_KEY is missing")
            sdk = OpenAI(
                api_key=key,
                base_url=config.get("user_base_url", "https://api.xiaomimimo.com/v1"),
                timeout=60.0,
                max_retries=0,
            )
        self.sdk = sdk

    def _record(self, row):
        with self.log_lock:
            self.requests += 1
            self.wait_seconds += row["limiter_wait_s"]
            if self.path:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
            if self.requests == 1 or self.requests % 20 == 0:
                print(
                    f"MIMO requests={self.requests} limiter_wait_sum_s={self.wait_seconds:.2f}",
                    flush=True,
                )

    @staticmethod
    def _retry_after(exc, attempt):
        response = getattr(exc, "response", None)
        raw = response.headers.get("retry-after") if response is not None else None
        if raw:
            try:
                return max(0.0, float(raw))
            except ValueError:
                try:
                    return max(
                        0.0,
                        (
                            parsedate_to_datetime(raw) - datetime.now(timezone.utc)
                        ).total_seconds(),
                    )
                except (TypeError, ValueError):
                    pass
        return 2**attempt + random.uniform(0, 0.5)

    def complete(self, messages, trajectory_id):
        # UTF-8 bytes plus generous per-message framing bound the text token count.
        reservation = len(
            json.dumps(messages, ensure_ascii=False).encode("utf-8")
        ) + 256 * len(messages)
        reservation += self.max_tokens
        with self.slots:
            for attempt in range(4):
                if self.fatal.is_set():
                    raise UserSimulatorAPIError("MiMo batch already aborted")
                ticket, waited = self.limiter.acquire(reservation)
                if self.fatal.is_set():
                    self.limiter.finish(ticket, 0)
                    raise UserSimulatorAPIError("MiMo batch already aborted")
                started = time.monotonic()
                wall_started = time.time()
                usage = None
                status = None
                retry = False
                failure = None
                error_type = error_reason = finish_reason = None
                content_chars = None
                response_received = False
                try:
                    response = self.sdk.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.7,
                        top_p=0.8,
                        max_completion_tokens=self.max_tokens,
                        stream=False,
                        extra_body={"thinking": {"type": "disabled"}},
                    )
                    response_received = True
                    status = 200
                    # MiMo documents usage as optional; it is accounting, not dialogue validity.
                    usage_object = getattr(response, "usage", None)
                    usage = (
                        usage_object.model_dump() if usage_object is not None else None
                    )
                    if usage is not None and any(
                        type(usage.get(key)) is not int or usage[key] < 0
                        for key in (
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                        )
                    ):
                        usage = None
                    choices = getattr(response, "choices", None)
                    if not choices:
                        raise MimoResponseError("missing_choices")
                    choice = choices[0]
                    finish_reason = getattr(choice, "finish_reason", None)
                    content = getattr(getattr(choice, "message", None), "content", None)
                    content_chars = len(content) if isinstance(content, str) else None
                    if finish_reason != "stop":
                        reason = (
                            "content_filter"
                            if finish_reason == "content_filter"
                            else "unexpected_finish_reason"
                        )
                        raise MimoResponseError(
                            reason,
                            retryable=finish_reason
                            in (
                                None,
                                "length",
                                "repetition_truncation",
                                "content_filter",
                            ),
                        )
                    if not isinstance(content, str) or not content.strip():
                        raise MimoResponseError("empty_or_nontext_content")
                except Exception as exc:  # noqa: BLE001 - sanitize SDK errors and abort the entire rollout batch
                    status = getattr(exc, "status_code", status)
                    retryable = status == 429 or (
                        isinstance(status, int) and status >= 500
                    )
                    retryable |= type(exc).__name__ in {
                        "APIConnectionError",
                        "APITimeoutError",
                    }
                    if isinstance(exc, MimoResponseError):
                        retryable = exc.retryable
                    error_type = type(exc).__name__
                    error_reason = (
                        exc.reason if isinstance(exc, MimoResponseError) else error_type
                    )
                    retry = retryable and attempt < 3
                    # Never include raw SDK exception text, request headers, or credentials.
                    failure = f"MiMo request failed: {error_type}, reason={error_reason}, status={status}, attempt={attempt + 1}"
                    if retry:
                        self.limiter.backoff(self._retry_after(exc, attempt))
                    else:
                        self.fatal.set()
                finally:
                    actual = usage.get("total_tokens") if usage else None
                    self.limiter.finish(
                        ticket, actual if isinstance(actual, int) else None
                    )
                    self._record(
                        {
                            "trajectory_id": trajectory_id,
                            "attempt": attempt + 1,
                            "started_at": wall_started,
                            "finished_at": time.time(),
                            "latency_s": time.monotonic() - started,
                            "limiter_wait_s": waited,
                            "status": status,
                            "retry": retry,
                            "success": failure is None,
                            "usage": usage,
                            "usage_missing": usage is None,
                            "response_received": response_received,
                            "finish_reason": finish_reason,
                            "content_chars": content_chars,
                            "error_type": error_type,
                            "error_reason": error_reason,
                            "reserved_tokens": reservation,
                        }
                    )
                if failure is None:
                    return content, usage
                if not retry:
                    raise UserSimulatorAPIError(failure) from None


def estimated_cost_usd(usage):
    """Published MiMo-v2.5 prices at 2026-09-19; estimate, not a billing receipt."""
    if usage is None:
        return None
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    prompt = usage.get("prompt_tokens", 0)
    return (
        (prompt - cached) * 0.14
        + cached * 0.0028
        + usage.get("completion_tokens", 0) * 0.28
    ) / 1_000_000
