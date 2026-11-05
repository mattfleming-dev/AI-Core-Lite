"""SQLite store: documents, chunks, vectors and a full-text index."""
import re
import sqlite3
import threading
import time
from array import array

try:
    import numpy as np
except ImportError:
    np = None


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.RLock()
        self.cache = {}
        with self.lock:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS collections(name TEXT PRIMARY KEY, embedder TEXT, dim INTEGER);
                CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, collection TEXT, source TEXT, sha TEXT,
                    chunks INTEGER, updated REAL, UNIQUE(collection, source));
                CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, doc_id INTEGER, collection TEXT,
                    idx INTEGER, text TEXT, emb BLOB);
                CREATE INDEX IF NOT EXISTS chunks_collection ON chunks(collection);
            """)
            try:
                self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='porter unicode61')")
                self.fts = True
            except sqlite3.OperationalError:
                self.fts = False

    def close(self):
        with self.lock:
            self.db.close()

    def ensure_collection(self, name, embedder_id, dim):
        with self.lock:
            row = self.db.execute("SELECT embedder, dim FROM collections WHERE name=?", (name,)).fetchone()
            if row is None:
                self.db.execute("INSERT INTO collections VALUES(?,?,?)", (name, embedder_id, dim))
            elif row[0] != embedder_id or row[1] != dim:
                raise ValueError(f"Collection '{name}' was built with embedder {row[0]}; "
                                 f"delete it or use the same embedding model ({embedder_id} requested)")

    def doc_sha(self, collection, source):
        with self.lock:
            row = self.db.execute("SELECT sha FROM docs WHERE collection=? AND source=?", (collection, source)).fetchone()
            return row[0] if row else None

    def _drop_doc(self, doc_id):
        ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
        if self.fts:
            self.db.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(i,) for i in ids])
        self.db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self.db.execute("DELETE FROM docs WHERE id=?", (doc_id,))

    def add_document(self, collection, source, sha, texts, vectors, embedder_id, dim):
        with self.lock:
            self.ensure_collection(collection, embedder_id, dim)
            old = self.db.execute("SELECT id FROM docs WHERE collection=? AND source=?", (collection, source)).fetchone()
            if old:
                self._drop_doc(old[0])
            doc_id = self.db.execute("INSERT INTO docs(collection, source, sha, chunks, updated) VALUES(?,?,?,?,?)",
                                     (collection, source, sha, len(texts), time.time())).lastrowid
            for index, (text, vector) in enumerate(zip(texts, vectors)):
                chunk_id = self.db.execute("INSERT INTO chunks(doc_id, collection, idx, text, emb) VALUES(?,?,?,?,?)",
                                           (doc_id, collection, index, text, array("f", vector).tobytes())).lastrowid
                if self.fts:
                    self.db.execute("INSERT INTO chunks_fts(rowid, text) VALUES(?,?)", (chunk_id, text))
            self.db.commit()
            self.cache.pop(collection, None)

    def delete_document(self, collection, source):
        with self.lock:
            row = self.db.execute("SELECT id FROM docs WHERE collection=? AND source=?", (collection, source)).fetchone()
            if row:
                self._drop_doc(row[0])
                self.db.commit()
                self.cache.pop(collection, None)
            return bool(row)

    def delete_collection(self, name):
        with self.lock:
            for (doc_id,) in self.db.execute("SELECT id FROM docs WHERE collection=?", (name,)).fetchall():
                self._drop_doc(doc_id)
            gone = self.db.execute("DELETE FROM collections WHERE name=?", (name,)).rowcount
            self.db.commit()
            self.cache.pop(name, None)
            return bool(gone)

    def collection_names(self):
        with self.lock:
            return [r[0] for r in self.db.execute("SELECT name FROM collections ORDER BY name")]

    def list_collections(self):
        with self.lock:
            rows = self.db.execute("""SELECT c.name, c.embedder, c.dim, COUNT(d.id), COALESCE(SUM(d.chunks), 0)
                                      FROM collections c LEFT JOIN docs d ON d.collection=c.name
                                      GROUP BY c.name ORDER BY c.name""").fetchall()
        return [{"name": n, "embedder": e, "dim": d, "documents": docs, "chunks": chunks} for n, e, d, docs, chunks in rows]

    def list_documents(self, collection):
        with self.lock:
            rows = self.db.execute("SELECT source, chunks, updated FROM docs WHERE collection=? ORDER BY source",
                                   (collection,)).fetchall()
        return [{"source": s, "chunks": c, "updated": u} for s, c, u in rows]

    def _matrix(self, collection):
        cached = self.cache.get(collection)
        if cached:
            return cached
        rows = self.db.execute("SELECT id, emb FROM chunks WHERE collection=?", (collection,)).fetchall()
        ids = [r[0] for r in rows]
        vectors = [array("f", r[1]) for r in rows]
        matrix = np.array(vectors, dtype="float32") if (np is not None and vectors) else vectors
        self.cache[collection] = (ids, matrix)
        return self.cache[collection]

    def vector_search(self, collections, embedder_id, query, limit):
        """Return [(chunk_id, cosine)] best first, over collections built with this embedder."""
        scored = []
        with self.lock:
            usable = [r[0] for r in self.db.execute(
                f"SELECT name FROM collections WHERE embedder=? AND name IN ({','.join('?' * len(collections))})",
                [embedder_id, *collections])] if collections else []
            for name in usable:
                ids, matrix = self._matrix(name)
                if not ids:
                    continue
                if np is not None:
                    sims = matrix @ np.array(query, dtype="float32")
                    scored.extend(zip(ids, (float(s) for s in sims)))
                else:
                    scored.extend((i, sum(a * b for a, b in zip(v, query))) for i, v in zip(ids, matrix))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def keyword_search(self, collections, query, limit):
        tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1][:24]
        if not (self.fts and tokens and collections):
            return []
        match = " OR ".join(f'"{t}"' for t in tokens)
        marks = ",".join("?" * len(collections))
        with self.lock:
            try:
                rows = self.db.execute(
                    f"""SELECT c.id FROM chunks_fts f JOIN chunks c ON c.id = f.rowid
                        WHERE chunks_fts MATCH ? AND c.collection IN ({marks})
                        ORDER BY bm25(chunks_fts) LIMIT ?""", [match, *collections, limit]).fetchall()
            except sqlite3.OperationalError:
                return []
        return [r[0] for r in rows]

    def get_chunks(self, ids):
        if not ids:
            return {}
        with self.lock:
            rows = self.db.execute(
                f"""SELECT c.id, c.collection, d.source, c.idx, c.text FROM chunks c JOIN docs d ON d.id=c.doc_id
                    WHERE c.id IN ({','.join('?' * len(ids))})""", list(ids)).fetchall()
        return {r[0]: {"id": r[0], "collection": r[1], "source": r[2], "index": r[3], "text": r[4]} for r in rows}
