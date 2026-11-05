"""Command line: serve, ingest, search, collections, delete."""
import argparse
import json
import logging
import sys

from . import config as cfg, ingest, retriever
from .embedder import make_embedder
from .server import serve
from .store import Store


def run_app(config, command, no_browser=False):
    """Background mode for desktop use: single instance, pid file, no console output."""
    import os
    import signal
    import socket
    import urllib.request
    import webbrowser
    from pathlib import Path

    url = f"http://{config['listen_host']}:{config['listen_port']}/"
    pid_file = Path(config["db_path"]).with_name("lite_rag.pid")
    if command == "stop":
        try:
            os.kill(int(pid_file.read_text()), signal.SIGTERM)
        except (OSError, ValueError):
            pass
        pid_file.unlink(missing_ok=True)
        return 0
    with socket.socket() as probe:
        if probe.connect_ex((config["listen_host"], config["listen_port"])) == 0:
            if not no_browser:
                webbrowser.open(url)
            return 0
    log_path = pid_file.with_name("lite_rag.log")
    logging.basicConfig(filename=str(log_path), level=logging.INFO, format="%(asctime)s %(message)s")
    app, server = serve(config)
    pid_file.write_text(str(os.getpid()))
    if not no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    finally:
        pid_file.unlink(missing_ok=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lite_rag", description="RAG wrapper for OpenAI/Anthropic compatible endpoints")
    parser.add_argument("--config", help="path to a JSON config file")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="start the wrapper and web UI")
    s.add_argument("--port", type=int)
    s.add_argument("--upstream")
    a = sub.add_parser("app", help="run in the background and open the UI (used by the desktop shortcut)")
    a.add_argument("--no-browser", action="store_true")
    sub.add_parser("stop", help="stop the background app")
    i = sub.add_parser("ingest", help="index files or folders")
    i.add_argument("collection")
    i.add_argument("paths", nargs="+")
    q = sub.add_parser("search", help="test retrieval")
    q.add_argument("collection")
    q.add_argument("query")
    sub.add_parser("collections", help="list collections")
    d = sub.add_parser("delete", help="delete a collection")
    d.add_argument("collection")
    args = parser.parse_args(argv)

    config = cfg.load(args.config)
    if args.cmd in ("app", "stop"):
        return run_app(config, args.cmd, getattr(args, "no_browser", False))
    if args.cmd == "serve":
        if args.port:
            config["listen_port"] = args.port
        if args.upstream:
            config["upstream_url"] = cfg.normalise_origin(args.upstream)
            if not config["embed"].get("model"):
                config["embed"]["url"] = config["upstream_url"]
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        app, server = serve(config)
        print(f"RAG wrapper ready: http://{config['listen_host']}:{config['listen_port']}/  (upstream {config['upstream_url']})")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Stopped.")
        return 0

    store, embedder = Store(config["db_path"]), make_embedder(config)
    if args.cmd == "ingest":
        for path in args.paths:
            print(path, json.dumps(ingest.ingest_path(store, embedder, config, args.collection, path)))
    elif args.cmd == "search":
        hits, note = retriever.retrieve(store, embedder, [args.collection], args.query, config["top_k"], config["min_similarity"])
        for n, hit in enumerate(hits, 1):
            print(f"[{n}] {hit['source']} (sim {hit['similarity']})\n    {hit['text'][:200]!r}")
        if note:
            print("note:", note)
    elif args.cmd == "collections":
        for c in store.list_collections():
            print(f"{c['name']}: {c['documents']} documents, {c['chunks']} chunks ({c['embedder']})")
    elif args.cmd == "delete":
        print("deleted" if store.delete_collection(args.collection) else "not found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
