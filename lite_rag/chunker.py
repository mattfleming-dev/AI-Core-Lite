"""Split text into overlapping chunks along natural boundaries."""

SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " "]


def _split(text, size, separators):
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i:i + size] for i in range(0, len(text), size)]
    out = []
    for part in text.split(separators[0]):
        if len(part) <= size:
            out.append(part)
        else:
            out.extend(_split(part, size, separators[1:]))
    return [p for p in out if p.strip()]


def split_text(text, size=900, overlap=120):
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    size = max(100, int(size))
    overlap = max(0, min(int(overlap), size // 2))
    chunks, current = [], ""
    for piece in _split(text, size, SEPARATORS):
        piece = piece.strip()
        if not piece:
            continue
        if current and len(current) + len(piece) + 1 > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            cut = tail.find(" ")
            tail = tail[cut + 1:] if cut >= 0 else tail
            current = f"{tail}\n{piece}" if tail else piece
        else:
            current = f"{current}\n{piece}" if current else piece
    if current.strip():
        chunks.append(current)
    return chunks
