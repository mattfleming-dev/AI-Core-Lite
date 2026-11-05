"""Index files and text into a collection."""
import hashlib
from pathlib import Path

from . import chunker, extract


def _index(store, embedder, config, collection, source, text):
    sha = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    if store.doc_sha(collection, source) == sha:
        return "skipped", 0
    pieces = chunker.split_text(text, config["chunk_size"], config["chunk_overlap"])
    if not pieces:
        return "skipped", 0
    vectors = embedder.embed(pieces)
    store.add_document(collection, source, sha, pieces, vectors, embedder.id, len(vectors[0]))
    return "indexed", len(pieces)


def ingest_text(store, embedder, config, collection, source, text):
    state, chunks = _index(store, embedder, config, collection, source, text)
    return {"indexed": int(state == "indexed"), "skipped": int(state == "skipped"), "chunks": chunks, "errors": []}


def ingest_path(store, embedder, config, collection, path, recursive=True):
    root = Path(path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    files = [root] if root.is_file() else sorted(p for p in (root.rglob("*") if recursive else root.glob("*")) if p.is_file())
    allowed = {e.lower() for e in config["extensions"]}
    result = {"indexed": 0, "skipped": 0, "chunks": 0, "errors": []}
    for file in files:
        if root.is_dir() and (file.suffix.lower() not in allowed or any(part.startswith(".") for part in file.relative_to(root).parts)):
            continue
        try:
            text = extract.read_text(file, config["max_file_bytes"])
            if text is None:
                result["skipped"] += 1
                continue
            state, chunks = _index(store, embedder, config, collection, str(file), text)
            result["indexed" if state == "indexed" else "skipped"] += 1
            result["chunks"] += chunks
        except Exception as error:
            result["errors"].append(f"{file}: {error}")
    return result
