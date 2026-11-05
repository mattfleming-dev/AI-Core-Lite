"""Hybrid retrieval (keyword + vector, reciprocal-rank fusion) and prompt injection."""
import logging

log = logging.getLogger("lite_rag")
RRF_K = 60

CONTEXT_PREAMBLE = ("Use the reference material below when it is relevant to the user's request. "
                    "Cite sources as [n]. If the material does not help, answer normally and do not invent sources.")


def retrieve(store, embedder, collections, query, top_k=5, min_similarity=0.2):
    pool = max(top_k * 4, 20)
    vector_hits, note = [], ""
    try:
        query_vector = embedder.embed([query])[0]
        vector_hits = store.vector_search(collections, embedder.id, query_vector, pool)
    except Exception as error:
        note = f"vector search unavailable: {error}"
        log.warning(note)
    keyword_hits = store.keyword_search(collections, query, pool)
    similarity = dict(vector_hits)
    scores = {}
    for rank, (chunk_id, _) in enumerate(vector_hits):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    for rank, chunk_id in enumerate(keyword_hits):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    keyword_set = set(keyword_hits)
    keep = [i for i in scores if i in keyword_set or similarity.get(i, 0.0) >= min_similarity]
    keep.sort(key=lambda i: scores[i], reverse=True)
    keep = keep[:top_k]
    chunks = store.get_chunks(keep)
    hits = []
    for chunk_id in keep:
        if chunk_id in chunks:
            hits.append({**chunks[chunk_id], "score": round(scores[chunk_id], 5),
                         "similarity": round(similarity.get(chunk_id, 0.0), 4)})
    return hits, note


def build_context(hits, max_chars=6000):
    blocks, used = [], 0
    for number, hit in enumerate(hits, 1):
        block = f"[{number}] {hit['source']}\n{hit['text']}"
        if blocks and used + len(block) > max_chars:
            break
        blocks.append(block[:max_chars])
        used += len(block)
    return CONTEXT_PREAMBLE + "\n\n" + "\n\n".join(blocks) if blocks else ""


def message_text(content):
    """Text of an OpenAI message, ignoring image and other non-text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", "input_text"))
    return ""


def last_user_query(messages, limit=2000):
    for message in reversed(messages):
        if message.get("role") == "user":
            text = message_text(message.get("content")).strip()
            if text:
                return text[-limit:]
    return ""


def inject_context(messages, context):
    """Return a copy of messages with the context in the system message. Image parts are untouched."""
    if not context:
        return messages
    messages = [dict(m) for m in messages]
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content")
        if isinstance(content, list):
            messages[0]["content"] = content + [{"type": "text", "text": context}]
        else:
            messages[0]["content"] = f"{content or ''}\n\n{context}".strip()
    else:
        messages.insert(0, {"role": "system", "content": context})
    return messages
