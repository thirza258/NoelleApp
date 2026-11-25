"""Dedicated QThread that owns the DEALER socket.

ZeroMQ sockets are not thread-safe, and ``QSocketNotifier`` on ``ZMQ_FD`` is
edge-triggered (easy to lose messages), so this thread polls the socket itself:
outgoing frames are queued from the GUI thread and sent here, incoming events
are decoded here and delivered to the GUI thread as queued Qt signals.
"""

from __future__ import annotations

import queue

import zmq
from PySide6.QtCore import QThread, Signal

from noelle.runtime.protocol import ProtocolError, decode


class StreamListener(QThread):
    message_received = Signal(dict)
    connection_error = Signal(str)

    def __init__(self, endpoint: str, parent: object | None = None) -> None:
        super().__init__(parent)
        self._endpoint = endpoint
        self._outgoing: queue.Queue[bytes] = queue.Queue()
        self._running = True

    def send(self, frame: bytes) -> None:
        if not self._running:
            return  # the socket is gone (or stopping); callers fail their pending requests instead of queueing forever
        self._outgoing.put(frame)

    def stop(self, timeout_ms: int = 2000) -> None:
        self._running = False
        self.wait(timeout_ms)

    def run(self) -> None:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVHWM, 0)
        sock.setsockopt(zmq.SNDHWM, 0)
        try:
            sock.connect(self._endpoint)
        except zmq.ZMQError as exc:
            self._running = False
            self.connection_error.emit(str(exc))
            sock.close(linger=0)
            return
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        try:
            while self._running:
                while True:
                    try:
                        frame = self._outgoing.get_nowait()
                    except queue.Empty:
                        break
                    sock.send(frame)
                ready = dict(poller.poll(15))
                if sock not in ready:
                    continue
                while True:
                    try:
                        data = sock.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        message = decode(data)
                    except ProtocolError:
                        continue
                    self.message_received.emit(message)
        except zmq.ZMQError as exc:
            self._running = False
            self.connection_error.emit(str(exc))
        finally:
            self._running = False
            sock.close(linger=0)
