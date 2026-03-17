from __future__ import annotations

import hashlib
import html
import re
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CrawlCandidate:
    text: str
    source_url: str


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        data = data.strip()
        if data:
            self.parts.append(data)

    def text(self) -> str:
        return " ".join(self.parts)


def canonical_text(text: str) -> str:
    return _WS_RE.sub(" ", text.strip())


def sentence_key(text: str) -> str:
    canon = canonical_text(text).lower()
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()


def is_high_quality_line(text: str) -> bool:
    t = canonical_text(text)
    low = t.lower()
    if len(t) < 20 or len(t) > 220:
        return False
    if len(re.findall(r"[A-Za-z]", t)) < 12:
        return False
    if "http://" in low or "https://" in low or "@" in t:
        return False
    if re.search(r"\b(edit|citation needed|isbn|retrieved from)\b", low):
        return False
    if re.search(r"[`{}<>]|\b(console|javascript|function|select\s+.+from)\b", low):
        return False
    words = re.findall(r"[a-z]+", low)
    if len(words) < 4:
        return False
    vowels = sum(ch in "aeiou" for ch in low)
    if vowels < max(6, len(low) // 18):
        return False
    return t[-1] in ".!?"


def extract_candidate_lines(html_text: str) -> list[str]:
    parser = _TextExtractor()
    parser.feed(html_text)
    text = html.unescape(parser.text())
    text = _TAG_RE.sub(" ", text)
    lines: list[str] = []
    for part in _SENTENCE_SPLIT_RE.split(text):
        candidate = canonical_text(part)
        if is_high_quality_line(candidate):
            lines.append(candidate)
    return lines


def fetch_url_text(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "SRM-Crawler/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def collect_new_lines(urls: list[str], known_lines: set[str] | None = None) -> list[CrawlCandidate]:
    seen = set(known_lines or set())
    out: list[CrawlCandidate] = []
    for url in urls:
        html_text = fetch_url_text(url)
        for line in extract_candidate_lines(html_text):
            key = sentence_key(line)
            if key in seen:
                continue
            seen.add(key)
            out.append(CrawlCandidate(text=line, source_url=url))
    return out
