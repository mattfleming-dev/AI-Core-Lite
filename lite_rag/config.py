"""Configuration for the RAG wrapper."""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

DEFAULTS = {
    "listen_host": "127.0.0.1",
    "listen_port": 8100,
    "upstream_url": "http://127.0.0.1:8080",
    "upstream_timeout": 1800,
    "db_path": "rag.sqlite",
    "default_collection": "default",
    "auto_rag": True,
    "top_k": 5,
    "min_similarity": 0.2,
    "max_context_chars": 6000,
    "chunk_size": 900,
    "chunk_overlap": 120,
    "max_file_bytes": 5_000_000,
    "admin_key": "",
    "embed": {"mode": "auto", "url": "", "model": "", "api_key": "", "batch": 16},
    "extensions": [".txt", ".md", ".rst", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml",
                   ".ini", ".cfg", ".csv", ".html", ".htm", ".xml", ".c", ".h", ".cpp", ".hpp",
                   ".cs", ".java", ".go", ".rs", ".ps1", ".bat", ".sh", ".sql", ".css", ".pdf"],
}


def normalise_origin(url):
    """Strip a trailing /v1 so client paths (which include /v1) can be forwarded unchanged."""
    url = (url or "").strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"Invalid upstream URL: {url!r}")
    return url


def load(path=None):
    config = json.loads(json.dumps(DEFAULTS))
    base = Path.cwd()
    if path:
        file = Path(path)
        base = file.resolve().parent
        user = json.loads(file.read_text(encoding="utf-8"))
        for key, value in user.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
    if os.environ.get("LITE_RAG_UPSTREAM"):
        config["upstream_url"] = os.environ["LITE_RAG_UPSTREAM"]
    config["upstream_url"] = normalise_origin(config["upstream_url"])
    if not config["embed"].get("url"):
        config["embed"]["url"] = config["upstream_url"]
    else:
        config["embed"]["url"] = normalise_origin(config["embed"]["url"])
    db = Path(config["db_path"])
    config["db_path"] = str(db if db.is_absolute() else base / db)
    return config
