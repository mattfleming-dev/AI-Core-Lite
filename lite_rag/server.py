"""HTTP server: OpenAI + Anthropic compatible proxy with retrieval, plus the admin API and web UI."""
import base64
import collections
import http.client
import json
import logging
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import __version__, anthropic, config as cfg, extract, ingest, retriever
from .embedder import make_embedder
from .store import Store

log = logging.getLogger("lite_rag")
UI_FILE = Path(__file__).parent / "ui" / "index.html"
MAX_BODY = 200 * 1024 * 1024
SETTING_KEYS = {"default_collection": str, "auto_rag": bool, "top_k": int, "min_similarity": float, "max_context_chars": int}


class App:
    def __init__(self, config):
        self.config = config
        self.store = Store(config["db_path"])
        self.embedder = make_embedder(config)
        self.activity = collections.deque(maxlen=40)
        self.settings_path = Path(config["db_path"] + ".settings.json")
        if self.settings_path.exists():
            try:
                self.config.update({k: v for k, v in json.loads(self.settings_path.read_text("utf-8")).items() if k in SETTING_KEYS})
            except (OSError, ValueError):
                log.warning("Ignoring unreadable settings file")

    def save_settings(self, values):
        for key, kind in SETTING_KEYS.items():
            if key in values:
                self.config[key] = kind(values[key])
        self.settings_path.write_text(json.dumps({k: self.config[k] for k in SETTING_KEYS}, indent=2), encoding="utf-8")

    def settings(self):
        return {k: self.config[k] for k in SETTING_KEYS}

    def is_loopback(self):
        return self.config["listen_host"] in ("127.0.0.1", "localhost", "::1")

    def pick_collections(self, rag, header):
        """Collections to search for this request, honouring per-request overrides."""
        if rag is False or (isinstance(rag, dict) and rag.get("enabled") is False):
            return []
        names = []
        if isinstance(rag, dict):
            names = rag.get("collections") or ([rag["collection"]] if rag.get("collection") else [])
        if not names and header:
            names = [n.strip() for n in header.split(",") if n.strip()]
        if not names and (self.config["auto_rag"] or rag):
            names = [self.config["default_collection"]]
        existing = set(self.store.collection_names())
        if names == ["*"]:
            return sorted(existing)
        return [n for n in names if n in existing]

    def apply_rag(self, messages, rag, header):
        """Return (messages, hits). Never raises: retrieval problems fall back to the original messages."""
        try:
            names = self.pick_collections(rag, header)
            query = retriever.last_user_query(messages)
            if not names or not query:
                return messages, []
            top_k = int(rag.get("top_k", self.config["top_k"])) if isinstance(rag, dict) else self.config["top_k"]
            hits, _ = retriever.retrieve(self.store, self.embedder, names, query, top_k, self.config["min_similarity"])
            context = retriever.build_context(hits, self.config["max_context_chars"])
            return retriever.inject_context(messages, context), hits if context else []
        except Exception:
            log.exception("RAG failed; sending request unchanged")
            return messages, []


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"
        server_version = f"lite-rag/{__version__}"

        def log_message(self, fmt, *args):
            log.debug(fmt, *args)

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Expose-Headers", "X-RAG-Sources")
            super().end_headers()

        # ---------- helpers ----------
        def read_body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("Request body too large")
            return self.rfile.read(length) if length else b""

        def read_json(self):
            raw = self.read_body()
            return json.loads(raw.decode("utf-8")) if raw else {}

        def send_json(self, status, payload, headers=None):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def is_anthropic(self):
            return bool(self.headers.get("anthropic-version"))

        def fail(self, status, message, anthropic_style=False):
            if anthropic_style:
                self.send_json(status, anthropic.error_body(message))
            else:
                self.send_json(status, {"error": {"message": message, "type": "gateway_error", "code": status}})

        def upstream_headers(self):
            headers = {"Content-Type": self.headers.get("Content-Type") or "application/json"}
            auth = self.headers.get("Authorization")
            if not auth and self.headers.get("x-api-key"):
                auth = "Bearer " + self.headers["x-api-key"]
            if auth:
                headers["Authorization"] = auth
            if self.headers.get("Accept"):
                headers["Accept"] = self.headers["Accept"]
            return headers

        def open_upstream(self, method, path, body=None):
            parts = urlsplit(app.config["upstream_url"])
            klass = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
            conn = klass(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80),
                         timeout=app.config["upstream_timeout"])
            conn.request(method, (parts.path.rstrip("/") + path), body, self.upstream_headers())
            return conn, conn.getresponse()

        def pump(self, response, extra=None):
            """Relay an upstream response unchanged, flushing as data arrives (works for SSE)."""
            self.send_response(response.status)
            for name in ("Content-Type", "Cache-Control", "Content-Length"):
                if response.getheader(name):
                    self.send_header(name, response.getheader(name))
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            while True:
                data = response.read1(65536)
                if not data:
                    break
                self.wfile.write(data)
                self.wfile.flush()

        def note(self, endpoint, model, hits, status):
            app.activity.appendleft({"time": time.time(), "endpoint": endpoint, "model": model or "",
                                     "sources": len(hits), "status": status})

        @staticmethod
        def sources_header(hits):
            if not hits:
                return {}
            payload = [{"n": i, "source": h["source"], "collection": h["collection"]} for i, h in enumerate(hits, 1)]
            return {"X-RAG-Sources": json.dumps(payload)[:3000]}

        # ---------- routing ----------
        def do_OPTIONS(self):
            self.send_response(204)
            self.end_headers()

        def do_GET(self):
            self.route("GET")

        def do_POST(self):
            self.route("POST")

        def do_DELETE(self):
            self.route("DELETE")

        def route(self, method):
            path = urlsplit(self.path).path
            try:
                if method == "GET" and path in ("/", "/ui", "/ui/"):
                    return self.serve_ui()
                if path == "/health":
                    return self.send_json(200, {"status": "ok", "version": __version__})
                if path.startswith("/rag/"):
                    return self.admin(method, path)
                if method == "POST" and path == "/v1/chat/completions":
                    return self.openai_chat()
                if method == "POST" and path == "/v1/messages":
                    return self.anthropic_chat()
                if method == "POST" and path == "/v1/messages/count_tokens":
                    body = self.read_json()
                    text = json.dumps(body.get("messages", [])) + json.dumps(body.get("system", ""))
                    return self.send_json(200, {"input_tokens": max(1, len(text) // 4)})
                if method == "GET" and path == "/v1/models" and self.is_anthropic():
                    return self.anthropic_models()
                return self.passthrough(method)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (OSError, http.client.HTTPException) as error:
                self.fail(502, f"Upstream {app.config['upstream_url']} unreachable: {error}", self.is_anthropic())
            except (ValueError, KeyError) as error:
                self.fail(400, str(error), self.is_anthropic())
            except Exception as error:
                log.exception("Unhandled error")
                self.fail(500, f"Internal error: {error}", self.is_anthropic())

        def serve_ui(self):
            data = UI_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def passthrough(self, method):
            body = self.read_body() if method == "POST" else None
            conn, response = self.open_upstream(method, self.path, body)
            try:
                self.pump(response)
            finally:
                conn.close()

        # ---------- OpenAI ----------
        def openai_chat(self):
            body = self.read_json()
            rag = body.pop("rag", None)
            hits = []
            if self.headers.get("X-RAG-Disable"):
                rag = False
            if isinstance(body.get("messages"), list):
                body["messages"], hits = app.apply_rag(body["messages"], rag, self.headers.get("X-RAG-Collection"))
            conn, response = self.open_upstream("POST", "/v1/chat/completions", json.dumps(body).encode("utf-8"))
            try:
                self.note("openai", body.get("model"), hits, response.status)
                self.pump(response, self.sources_header(hits))
            finally:
                conn.close()

        # ---------- Anthropic ----------
        def anthropic_models(self):
            conn, response = self.open_upstream("GET", "/v1/models")
            try:
                data = json.loads(response.read().decode("utf-8"))
            finally:
                conn.close()
            self.send_json(200, anthropic.models_to_anthropic(data))

        def anthropic_chat(self):
            body = self.read_json()
            rag = body.pop("rag", None)
            model = body.get("model", "")
            request = anthropic.to_openai_request(body)
            if self.headers.get("X-RAG-Disable"):
                rag = False
            request["messages"], hits = app.apply_rag(request["messages"], rag, self.headers.get("X-RAG-Collection"))
            conn, response = self.open_upstream("POST", "/v1/chat/completions", json.dumps(request).encode("utf-8"))
            try:
                self.note("anthropic", model, hits, response.status)
                extra = self.sources_header(hits)
                if response.status >= 400:
                    detail = response.read().decode("utf-8", errors="replace")[:500]
                    return self.send_json(response.status, anthropic.error_body(detail or f"Upstream HTTP {response.status}"), extra)
                if not request["stream"]:
                    data = json.loads(response.read().decode("utf-8"))
                    return self.send_json(200, anthropic.to_anthropic_response(data, model), extra)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                for key, value in extra.items():
                    self.send_header(key, value)
                self.end_headers()

                def chunks():
                    while True:
                        line = response.readline()
                        if not line:
                            return
                        line = line.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            return
                        try:
                            yield json.loads(payload)
                        except ValueError:
                            continue

                for event, data in anthropic.stream_to_anthropic(chunks(), model):
                    self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8"))
                    self.wfile.flush()
            finally:
                conn.close()

        # ---------- admin API ----------
        def authorised(self):
            key = app.config["admin_key"]
            return not key or self.headers.get("Authorization") == f"Bearer {key}"

        def admin(self, method, path):
            if not self.authorised():
                return self.fail(401, "Admin key required")
            query = parse_qs(urlsplit(self.path).query)
            if method == "GET" and path == "/rag/status":
                return self.send_json(200, self.status())
            if method == "GET" and path == "/rag/collections":
                return self.send_json(200, {"collections": app.store.list_collections()})
            if method == "GET" and path == "/rag/documents":
                return self.send_json(200, {"documents": app.store.list_documents((query.get("collection") or ["default"])[0])})
            if method == "GET" and path == "/rag/activity":
                return self.send_json(200, {"activity": list(app.activity)})
            if method == "DELETE" and path.startswith("/rag/collections/"):
                name = path.rsplit("/", 1)[1]
                return self.send_json(200, {"deleted": app.store.delete_collection(name)})
            if method != "POST":
                return self.fail(404, "Unknown admin endpoint")
            body = self.read_json()
            if path == "/rag/settings":
                app.save_settings(body)
                return self.send_json(200, app.settings())
            if path == "/rag/search":
                names = body.get("collections") or [body.get("collection") or app.config["default_collection"]]
                hits, note = retriever.retrieve(app.store, app.embedder, names, body.get("query", ""),
                                                int(body.get("top_k", app.config["top_k"])), app.config["min_similarity"])
                return self.send_json(200, {"hits": hits, "note": note})
            if path == "/rag/documents/delete":
                return self.send_json(200, {"deleted": app.store.delete_document(body["collection"], body["source"])})
            collection = (body.get("collection") or app.config["default_collection"]).strip()
            if not collection:
                raise ValueError("collection is required")
            if path == "/rag/ingest":
                if body.get("text") is not None:
                    result = ingest.ingest_text(app.store, app.embedder, app.config, collection, body.get("source") or "pasted-text", body["text"])
                else:
                    if not (app.is_loopback() or app.config["admin_key"]):
                        raise ValueError("Path ingest needs a loopback listener or an admin_key")
                    result = ingest.ingest_path(app.store, app.embedder, app.config, collection, body["path"], body.get("recursive", True))
                return self.send_json(200, result)
            if path == "/rag/upload":
                name = Path(body["filename"]).name
                raw = base64.b64decode(body["data_b64"])
                with tempfile.TemporaryDirectory() as folder:
                    target = Path(folder) / name
                    target.write_bytes(raw)
                    text = extract.read_text(target, app.config["max_file_bytes"])
                if text is None:
                    raise ValueError(f"{name}: unreadable, empty or too large")
                return self.send_json(200, ingest.ingest_text(app.store, app.embedder, app.config, collection, name, text))
            return self.fail(404, "Unknown admin endpoint")

        def status(self):
            upstream_ok, models = False, []
            try:
                conn, response = self.open_upstream("GET", "/v1/models")
                if response.status < 400:
                    upstream_ok = True
                    models = [m.get("id") for m in json.loads(response.read().decode("utf-8")).get("data", [])]
                conn.close()
            except Exception:
                pass
            return {"version": __version__, "upstream": app.config["upstream_url"], "upstream_ok": upstream_ok,
                    "models": models, "embedder": app.embedder.id, "keyword_search": app.store.fts,
                    "settings": app.settings(), "collections": app.store.list_collections(),
                    "endpoints": {"openai": f"http://{app.config['listen_host']}:{app.config['listen_port']}/v1",
                                  "anthropic": f"http://{app.config['listen_host']}:{app.config['listen_port']}"}}

    return Handler


def serve(config):
    app = App(config)
    server = ThreadingHTTPServer((config["listen_host"], config["listen_port"]), make_handler(app))
    server.daemon_threads = True
    return app, server
