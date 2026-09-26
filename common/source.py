"""
TransactionSource: the single interface every part of the ML/backend system
depends on for getting transactions.

WHY this exists (this is the most important file in the project):
every downstream piece — the FL clients, the inference service, the API —
should never need to know or care whether a transaction came from our fake
generator, a real company's SQL database, or a partner's webhook. They only
ever talk to a `TransactionSource`. That means swapping from synthetic data
to a real company's data later is a ONE LINE config change, not a rewrite.

This is the Dependency Inversion Principle applied to a data pipeline:
high-level code (the ML/API layer) depends on an abstraction, not on a
concrete data technology.
"""

from abc import ABC, abstractmethod
from typing import Iterator
import json
import time

import redis

from .models import Transaction


class TransactionSource(ABC):
    @abstractmethod
    def stream(self) -> Iterator[Transaction]:
        """Yield transactions forever (or until the source is exhausted)."""
        raise NotImplementedError


class SyntheticSource(TransactionSource):
    """
    Reads transactions pushed by generator/simulate.py onto a Redis Stream.
    This is our stand-in for a real live transaction feed during development.
    """

    def __init__(self, redis_host="localhost", redis_port=6379,
                 stream_name="transactions", block_ms=5000):
        self.client = redis.Redis(host=redis_host, port=redis_port,
                                   decode_responses=True)
        self.stream_name = stream_name
        self.block_ms = block_ms
        # "$" means "only new messages from now on" — start at the live tail
        self.last_id = "$"

    def stream(self) -> Iterator[Transaction]:
        while True:
            # XREAD blocks (waits) up to block_ms for new stream entries,
            # instead of polling in a tight loop — cheaper and more
            # realistic for how a real event stream consumer behaves.
            response = self.client.xread(
                {self.stream_name: self.last_id}, block=self.block_ms, count=10
            )
            if not response:
                continue
            _, entries = response[0]
            for entry_id, fields in entries:
                self.last_id = entry_id
                yield Transaction.from_json(fields["data"])


class SQLSource(TransactionSource):
    """
    STUB for swapping in a real company's database later.

    Real implementation would typically poll something like:
        SELECT * FROM transactions WHERE updated_at > :last_seen
        ORDER BY updated_at ASC
    on an interval, or better, hook into the DB's Change Data Capture (CDC)
    feed (e.g. Debezium) so new rows are pushed rather than polled.

    Left unimplemented on purpose — fill this in when you have real
    connection details, and nothing else in the system needs to change.
    """

    def __init__(self, connection_string: str, poll_interval_s: int = 5):
        self.connection_string = connection_string
        self.poll_interval_s = poll_interval_s

    def stream(self) -> Iterator[Transaction]:
        raise NotImplementedError(
            "Plug in a real DB connection here when you have one."
        )


class APISource(TransactionSource):
    """
    STUB for a company that exposes transactions via a REST endpoint or
    webhook instead of direct DB access. Same idea as SQLSource — fill in
    when needed, rest of the system is unaffected.
    """

    def __init__(self, api_url: str, api_key: str = None):
        self.api_url = api_url
        self.api_key = api_key

    def stream(self) -> Iterator[Transaction]:
        raise NotImplementedError(
            "Plug in real API polling/webhook handling here when you have it."
        )
