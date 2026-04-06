from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> List[str]:
    lowered = (text or "").strip().lower()
    if not lowered:
        return []

    tokens = _TOKEN_RE.findall(lowered)
    expanded: List[str] = list(tokens)

    for token in tokens:
        if "_" in token:
            expanded.extend(part for part in token.split("_") if part)

    return expanded


class BM25Index:
    def __init__(self, corpus_tokens: Iterable[Iterable[str]], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = []
        self.doc_lengths = []
        self.idf = {}

        corpus_list = [list(tokens) for tokens in corpus_tokens]
        self.corpus_size = len(corpus_list)

        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            return

        nd = Counter()
        total_len = 0

        for tokens in corpus_list:
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_len += doc_len

            for token in freqs:
                nd[token] += 1

        self.avg_doc_len = total_len / self.corpus_size
        self.idf = {
            token: math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for token, freq in nd.items()
        }

    def score(self, query_tokens: Iterable[str]) -> List[float]:
        query = list(query_tokens)
        if self.corpus_size == 0 or not query:
            return [0.0] * self.corpus_size

        scores = [0.0] * self.corpus_size

        for idx, freqs in enumerate(self.doc_freqs):
            doc_len = self.doc_lengths[idx]
            norm = self.k1 * (1.0 - self.b + self.b * doc_len / (self.avg_doc_len or 1.0))
            total = 0.0

            for token in query:
                tf = freqs.get(token, 0)
                if tf <= 0:
                    continue

                idf = self.idf.get(token, 0.0)
                total += idf * (tf * (self.k1 + 1.0)) / (tf + norm)

            scores[idx] = total

        return scores