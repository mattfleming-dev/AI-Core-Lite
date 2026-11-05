import base64
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lite_rag import chunker, config as cfg, retriever
from lite_rag.server import serve
from lite_rag.embedder import HashEmbedder
from lite_rag.store import Store

SEEN = []


class FakeUpstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _send(self, payload, status=200):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self._send({"object": "list", "data": [{"id": "fake-model", "object": "model"}]})
        self._send({"error": "nope"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        SEEN.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
        if self.path == "/v1/chat/completions":
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for piece in ("Hel", "lo"):
                    chunk = {"id": "c1", "model": "fake-model", "choices": [{"index": 0, "delta": {"content": piece}}]}
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                done = {"id": "c1", "model": "fake-model", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                self.wfile.write(b"data: " + json.dumps(done).encode() + b"\n\ndata: [DONE]\n\n")
                return
            return self._send({"id": "c1", "model": "fake-model", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1}})
        self._send({"ok": True})


def post(url, body, headers=None):
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, response.read(), response.headers


class Base(unittest.TestCase):
    admin_key = ""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.up = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
        threading.Thread(target=cls.up.serve_forever, daemon=True).start()
        config = cfg.load(None)
        config.update(listen_port=0, upstream_url=f"http://127.0.0.1:{cls.up.server_port}",
                      db_path=os.path.join(cls.tmp.name, "rag.db"), admin_key=cls.admin_key)
        cls.app, cls.srv = serve(config)
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.up.shutdown()
        cls.srv.server_close()
        cls.up.server_close()
        cls.app.store.close()
        cls.tmp.cleanup()


class CoreTests(unittest.TestCase):
    def test_chunker(self):
        text = "\n\n".join(f"Paragraph {i}. " + "word " * 80 for i in range(10))
        parts = chunker.split_text(text, 500, 50)
        self.assertGreater(len(parts), 5)
        self.assertTrue(all(len(p) <= 700 for p in parts))
        self.assertEqual(chunker.split_text("   ", 500, 50), [])

    def test_inject_preserves_images(self):
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        messages = [{"role": "user", "content": [{"type": "text", "text": "what is this"}, image]}]
        self.assertEqual(retriever.last_user_query(messages), "what is this")
        out = retriever.inject_context(messages, "CTX")
        self.assertEqual(out[0]["role"], "system")
        self.assertIn("CTX", out[0]["content"])
        self.assertIn(image, out[1]["content"])

    def test_store_roundtrip_and_skip(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(os.path.join(d, "s.db"))
            emb = HashEmbedder()
            from lite_rag import ingest
            config = cfg.load(None)
            first = ingest.ingest_text(store, emb, config, "docs", "a.txt", "The mitochondria is the powerhouse of the cell.")
            again = ingest.ingest_text(store, emb, config, "docs", "a.txt", "The mitochondria is the powerhouse of the cell.")
            self.assertEqual((first["indexed"], again["skipped"]), (1, 1))
            hits, _ = retriever.retrieve(store, emb, ["docs"], "powerhouse of the cell")
            self.assertEqual(hits[0]["source"], "a.txt")
            self.assertTrue(store.delete_document("docs", "a.txt"))
            store.close()


class ServerTests(Base):
    def setUp(self):
        SEEN.clear()
        post(self.base + "/rag/ingest", {"collection": "default", "source": "pets.txt",
                                          "text": "Our company mascot is a purple axolotl named Zorbo."})

    def upstream_messages(self):
        return [s for s in SEEN if s["path"] == "/v1/chat/completions"][-1]["body"]["messages"]

    def test_health_status_and_ui(self):
        with urllib.request.urlopen(self.base + "/rag/status") as r:
            status = json.loads(r.read())
        self.assertTrue(status["upstream_ok"])
        self.assertEqual(status["models"], ["fake-model"])
        with urllib.request.urlopen(self.base + "/") as r:
            self.assertIn(b"Lite RAG", r.read())

    def test_openai_injects_context(self):
        code, raw, headers = post(self.base + "/v1/chat/completions",
                                  {"model": "m", "messages": [{"role": "user", "content": "Who is the company mascot?"}]})
        self.assertEqual(json.loads(raw)["choices"][0]["message"]["content"], "Hello")
        self.assertIn("Zorbo", json.dumps(self.upstream_messages()))
        self.assertIn("pets.txt", headers.get("X-RAG-Sources", ""))

    def test_vision_request_passes_through(self):
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        post(self.base + "/v1/chat/completions", {"model": "m", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "mascot in this picture?"}, image]}]})
        sent = self.upstream_messages()
        self.assertIn(image, sent[-1]["content"])
        self.assertIn("Zorbo", json.dumps(sent))

    def test_rag_can_be_disabled(self):
        post(self.base + "/v1/chat/completions", {"model": "m", "rag": False,
                                                   "messages": [{"role": "user", "content": "Who is the company mascot?"}]})
        self.assertNotIn("Zorbo", json.dumps(self.upstream_messages()))
        self.assertNotIn("rag", [s for s in SEEN if s["path"] == "/v1/chat/completions"][-1]["body"])
        post(self.base + "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "Who is the company mascot?"}]},
             {"X-RAG-Disable": "1"})
        self.assertNotIn("Zorbo", json.dumps(self.upstream_messages()))

    def test_openai_streaming(self):
        code, raw, _ = post(self.base + "/v1/chat/completions",
                            {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
        self.assertIn(b"[DONE]", raw)
        self.assertIn(b"Hel", raw)

    def test_anthropic_non_streaming(self):
        code, raw, _ = post(self.base + "/v1/messages", {
            "model": "m", "max_tokens": 50, "system": "Be brief.",
            "messages": [{"role": "user", "content": "Who is the company mascot?"}]},
            {"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"})
        body = json.loads(raw)
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["content"][0]["text"], "Hello")
        self.assertEqual(body["stop_reason"], "end_turn")
        seen = [s for s in SEEN if s["path"] == "/v1/chat/completions"][-1]
        self.assertEqual(seen["auth"], "Bearer sk-ant-test")
        self.assertIn("Be brief.", json.dumps(seen["body"]["messages"]))
        self.assertIn("Zorbo", json.dumps(seen["body"]["messages"]))

    def test_anthropic_image_translation(self):
        post(self.base + "/v1/messages", {"model": "m", "max_tokens": 50, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]},
            {"anthropic-version": "2023-06-01"})
        parts = self.upstream_messages()[-1]["content"]
        urls = [p["image_url"]["url"] for p in parts if p.get("type") == "image_url"]
        self.assertEqual(urls, ["data:image/png;base64,AAAA"])

    def test_anthropic_streaming(self):
        code, raw, _ = post(self.base + "/v1/messages", {"model": "m", "max_tokens": 50, "stream": True,
                            "messages": [{"role": "user", "content": "hi"}]}, {"anthropic-version": "2023-06-01"})
        text = raw.decode()
        for event in ("message_start", "content_block_delta", "message_delta", "message_stop"):
            self.assertIn("event: " + event, text)

    def test_anthropic_tools_translated(self):
        post(self.base + "/v1/messages", {"model": "m", "max_tokens": 50,
             "tools": [{"name": "get_weather", "description": "w", "input_schema": {"type": "object", "properties": {}}}],
             "messages": [{"role": "user", "content": "hi"}]}, {"anthropic-version": "2023-06-01"})
        body = [s for s in SEEN if s["path"] == "/v1/chat/completions"][-1]["body"]
        self.assertEqual(body["tools"][0]["function"]["name"], "get_weather")

    def test_upload_and_delete(self):
        data = base64.b64encode(b"The launch code word is pineapple.").decode()
        code, raw, _ = post(self.base + "/rag/upload", {"collection": "up", "filename": "notes.md", "data_b64": data})
        self.assertEqual(json.loads(raw)["indexed"], 1)
        code, raw, _ = post(self.base + "/rag/search", {"collection": "up", "query": "launch code word"})
        self.assertEqual(json.loads(raw)["hits"][0]["source"], "notes.md")
        request = urllib.request.Request(self.base + "/rag/collections/up", method="DELETE")
        with urllib.request.urlopen(request) as r:
            self.assertTrue(json.loads(r.read())["deleted"])

    def test_passthrough_models(self):
        with urllib.request.urlopen(self.base + "/v1/models") as r:
            self.assertEqual(json.loads(r.read())["data"][0]["id"], "fake-model")


class AuthTests(Base):
    admin_key = "secret"

    def test_admin_requires_key(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.base + "/rag/status")
        self.assertEqual(ctx.exception.code, 401)
        request = urllib.request.Request(self.base + "/rag/status", headers={"Authorization": "Bearer secret"})
        with urllib.request.urlopen(request) as r:
            self.assertEqual(r.status, 200)


if __name__ == "__main__":
    unittest.main()
