"""Smaller serve-side robustness fixes of round 0213 (each test fails on the base a3a10b90a):

#3443 ConvCacheLog.poll without a lock; #4271 a streaming client that stops reading.  (MoMoE: upstream's tests of the
fixes it did not take - #4792, #4738, #5044 - and their fake batch engine are left out.)
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve.server import StrataEngine


class ConvCachePoll(unittest.TestCase):
    """ConvCacheLog.poll ran outside any lock: a second poll that moved `pos` between the first one's size check and
    its read left a negative read length (ValueError).  Found by HWSTALL while triaging #3443."""
    PARK = ("strata serve: conversation cache: parked 5000 tokens in 12.0 ms; parked=1 bytes=104857600 evictions=0 "
            "snapshot_bytes=104857600 reused_kv_bytes=0\n")

    def test_a_poll_that_overlaps_another_one_does_not_raise(self):
        import os
        from serve.server import ConvCacheLog
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(self.PARK, encoding="utf-8")
            c = ConvCacheLog()
            c.poll(str(log), 0)
            with open(log, "a", encoding="utf-8") as f:
                f.write(self.PARK)
            first_sized, first_may_go = threading.Event(), threading.Event()
            real_size = os.path.getsize
            calls = []

            def slow_first(path):
                size = real_size(path)
                if not calls and path == str(log):
                    calls.append(1)
                    first_sized.set()
                    first_may_go.wait(2.0)             # the first poll has its size; a second poll runs meanwhile
                return size
            errors = []

            def first():
                try:
                    c.poll(str(log), 0)
                except BaseException as e:             # noqa: BLE001
                    errors.append(e)
            with mock.patch("serve.server.os.path.getsize", slow_first):
                t = threading.Thread(target=first, daemon=True)
                t.start()
                self.assertTrue(first_sized.wait(5))
                with open(log, "a", encoding="utf-8") as f:
                    f.write(self.PARK * 2)
                second = threading.Thread(target=lambda: c.poll(str(log), 0), daemon=True)
                second.start()
                second.join(0.5)                       # (with the lock it waits for the first one)
                first_may_go.set()
                t.join(5)
                second.join(5)
            self.assertEqual(errors, [])
            self.assertEqual(c.poll(str(log), 0)["parks"], 4, "every parked line counted once")


class StalledClient(unittest.TestCase):
    """#4271: a streaming client that stops reading, without closing its connection, must not hold the engine's turn
    for ever: the write that makes no progress ends the request as a disconnect and the next request is served."""

    def test_a_client_that_stops_reading_gives_the_turn_back(self):
        import json
        import socket
        import time
        import urllib.request
        import serve.server as server
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, serve
        tok = ByteTokenizer()
        engine = MockEngine(tok, " ".join(str(i) for i in range(80000)), max_context=2_000_000)   # (not one repeated token)
        svc = Service(engine, tok, ChatTemplate(Path(server.__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        port = httpd.server_address[1]
        stuck = socket.socket()
        try:
            with mock.patch.object(server, "STREAM_WRITE_TIMEOUT_S", 0.5, create=True):
                stuck.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
                stuck.connect(("127.0.0.1", port))
                body = json.dumps({"messages": [{"role": "user", "content": "go"}], "stream": True,
                                   "max_tokens": 400000, "reasoning_effort": "none"}).encode()
                stuck.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                              b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
                # ... and it never reads.  Wait until the server is writing into a full buffer, then until it lets go.
                end = time.monotonic() + 3
                while time.monotonic() < end and not svc.fifo.locked():
                    time.sleep(0.01)
                self.assertTrue(svc.fifo.locked(), "the stuck client's request holds the engine's turn")
                end = time.monotonic() + 20
                while time.monotonic() < end and svc.fifo.locked():
                    time.sleep(0.05)
                self.assertFalse(svc.fifo.locked(), "the turn is still held by a client that does not read")
        finally:
            stuck.close()
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
