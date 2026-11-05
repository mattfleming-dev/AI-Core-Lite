"""Text embedders: an OpenAI-compatible /v1/embeddings client and a no-model hashing fallback."""
import json
import math
import re
import urllib.error
import urllib.request
import zlib


def normalise(vector):
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class HashEmbedder:
    """Feature-hashed bag of words and bigrams. Weak semantically, but needs no model."""
    dim = 512
    id = "hash-512"

    def embed(self, texts):
        out = []
        for text in texts:
            vector = [0.0] * self.dim
            words = re.findall(r"\w+", text.lower())
            for token in words + [f"{a}_{b}" for a, b in zip(words, words[1:])]:
                h = zlib.crc32(token.encode("utf-8"))
                vector[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
            out.append(normalise(vector))
        return out


class RemoteEmbedder:
    def __init__(self, base_url, model, api_key="", batch=16, timeout=120):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.batch, self.timeout = max(1, int(batch)), timeout
        self.id = f"remote:{model or 'default'}"

    def _post(self, texts):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"input": texts}
        if self.model:
            payload["model"] = self.model
        request = urllib.request.Request(self.base_url + "/v1/embeddings", json.dumps(payload).encode("utf-8"),
                                         headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Embedding request failed: HTTP {error.code} {error.read()[:200]!r}")
        except (urllib.error.URLError, OSError) as error:
            raise RuntimeError(f"Embedding endpoint unreachable: {error}")
        items = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        if len(items) != len(texts):
            raise RuntimeError("Embedding endpoint returned the wrong number of vectors")
        return [normalise(item["embedding"]) for item in items]

    def embed(self, texts):
        out = []
        for start in range(0, len(texts), self.batch):
            out.extend(self._post(texts[start:start + self.batch]))
        return out


def make_embedder(config):
    embed = config["embed"]
    mode = embed.get("mode", "auto")
    if mode == "hash" or (mode == "auto" and not embed.get("model")):
        return HashEmbedder()
    return RemoteEmbedder(embed["url"], embed.get("model", ""), embed.get("api_key", ""), embed.get("batch", 16))
