"""Read text out of files for indexing."""
import re
from html.parser import HTMLParser
from pathlib import Path


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self.skip:
            self.skip -= 1
        elif tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "section"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(markup):
    parser = _Text()
    parser.feed(markup)
    return re.sub(r"\n\s*\n+", "\n\n", "".join(parser.parts)).strip()


def read_pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("PDF support needs pypdf (pip install pypdf)")
    return "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)


def read_text(path, max_bytes=5_000_000):
    """Return the file's text, or None when it is binary, empty or too large."""
    path = Path(path)
    if path.stat().st_size > max_bytes:
        return None
    if path.suffix.lower() == ".pdf":
        return read_pdf(path) or None
    data = path.read_bytes()
    if b"\0" in data[:4096]:
        return None
    text = data.decode("utf-8", errors="replace")
    if path.suffix.lower() in (".html", ".htm"):
        text = html_to_text(text)
    return text if text.strip() else None
