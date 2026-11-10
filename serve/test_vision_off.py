"""serve/test_vision_off.py - an image encoder that does not start no longer ends the server: the model starts for text
only, and a picture request is refused with the reason.  On a Mac before macOS 14 the GPU encoder stopped at its first
picture (Metal has no BF16 there: kernel_mul_mm_bf16_f32 missing), and with it no model started at all."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import serve.server as server  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402

PICTURE = [{"role": "user", "content": [{"type": "text", "text": "what is it?"},
                                        {"type": "image", "source": "x.png"}]}]
CRASH = ("the vision encoder did not start: it ended without saying anything (killed by signal 6 (SIGABRT, Abort trap: "
         "6); last lines of its log: mtmd_encode_chunk + 24)")


class VisionOff(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(server, "VISION_OFF", "")
        p.start()
        self.addCleanup(p.stop)

    def service(self):
        tok = ByteTokenizer()
        return Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"),
                       vision=None)

    def test_an_encoder_that_does_not_start_leaves_text_only(self):
        with mock.patch.object(server, "Vision", side_effect=RuntimeError(CRASH)):
            self.assertIsNone(server.start_vision({"exe": "x"}, None, {}, False))
        self.assertEqual(server.VISION_OFF, CRASH)

    def test_a_missing_encoder_program_too(self):
        with mock.patch.object(server, "Vision", side_effect=FileNotFoundError("momoe-vision")):
            self.assertIsNone(server.start_vision({"exe": "x"}, None, {}, False))
        self.assertTrue(server.VISION_OFF.startswith("the image encoder did not start: "))

    def test_an_encoder_that_starts_is_kept(self):
        with mock.patch.object(server, "Vision", return_value="encoder"):
            self.assertEqual(server.start_vision({"exe": "x"}, None, {}, False), "encoder")
        self.assertEqual(server.VISION_OFF, "")

    def test_a_picture_request_says_why(self):
        with mock.patch.object(server, "Vision", side_effect=RuntimeError(CRASH)):
            server.start_vision({"exe": "x"}, None, {}, False)
        with self.assertRaisesRegex(ValueError, "pictures are off: the vision encoder did not start"):
            self.service().prepare(PICTURE, None, {})

    def test_a_server_set_up_without_pictures_says_so_as_before(self):
        with self.assertRaisesRegex(ValueError, "started without the vision encoder"):
            self.service().prepare(PICTURE, None, {})

    def test_text_still_works(self):
        with mock.patch.object(server, "Vision", side_effect=RuntimeError(CRASH)):
            server.start_vision({"exe": "x"}, None, {}, False)
        ids, _, _ = self.service().prepare([{"role": "user", "content": "hi"}], None, {})
        self.assertTrue(ids)


if __name__ == "__main__":
    unittest.main()
