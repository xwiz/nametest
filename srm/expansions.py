"""
srm/expansions.py — automatic query expansion using WordNet synonyms.

This module provides functions to auto-generate expansion dictionaries
from a knowledge base using NLTK WordNet, eliminating the need for
manual EXPANSIONS maintenance.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

try:
    import nltk
    from nltk.corpus import wordnet as wn
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


def ensure_nltk_data() -> None:
    """Download required NLTK WordNet data if not already present."""
    if not NLTK_AVAILABLE:
        raise ImportError("NLTK is required for auto-expansion. Install with: pip install nltk")
    try:
        wn.synsets('test')
    except LookupError:
        nltk.download('wordnet', quiet=True)


def get_wordnet_synonyms(word: str, max_synonyms: int = 5) -> List[str]:
    """
    Get synonyms for a word using WordNet.

    Args:
        word: The word to find synonyms for (lowercase)
        max_synonyms: Maximum number of synonyms to return

    Returns:
        List of synonym strings (lowercase), excluding the original word
    """
    if not NLTK_AVAILABLE:
        return []

    synonyms = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            syn = lemma.name().lower().replace('_', ' ')
            if syn != word and syn.isalpha():
                synonyms.add(syn)
            if len(synonyms) >= max_synonyms:
                break
        if len(synonyms) >= max_synonyms:
            break

    return sorted(list(synonyms))[:max_synonyms]


def build_expansions_from_kb(
    texts: List[str],
    min_document_freq: int = 2,
    max_terms: int = 200,
    max_synonyms_per_term: int = 3,
) -> Dict[str, List[str]]:
    """
    Build an EXPANSIONS dictionary from a knowledge base using WordNet.

    This analyzes the KB to find important terms (those appearing in
    multiple documents) and generates synonym expansions for them.

    Args:
        texts: List of memory texts from the KB
        min_document_freq: Minimum number of documents a term must appear in
        max_terms: Maximum number of terms to generate expansions for
        max_synonyms_per_term: Maximum synonyms per term

    Returns:
        Dictionary mapping term -> list of expansion terms
    """
    if not NLTK_AVAILABLE:
        return {}

    ensure_nltk_data()

    from .nlp import tokenise

    # Count document frequency for each term
    doc_freq: Dict[str, int] = defaultdict(int)
    for text in texts:
        tokens = set(tokenise(text))
        for token in tokens:
            doc_freq[token] += 1

    # Filter to important terms (appears in multiple docs)
    important_terms = [
        term for term, freq in sorted(doc_freq.items(), key=lambda x: -x[1])
        if freq >= min_document_freq and len(term) > 2
    ][:max_terms]

    # Generate expansions using WordNet
    expansions: Dict[str, List[str]] = {}
    for term in important_terms:
        synonyms = get_wordnet_synonyms(term, max_synonyms_per_term)
        if synonyms:
            expansions[term] = synonyms

    return expansions


def merge_expansions(
    manual: Dict[str, List[str]],
    auto: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Merge manual and auto-generated expansions, preferring manual ones.

    Args:
        manual: Manually curated EXPANSIONS dictionary
        auto: Auto-generated expansions from build_expansions_from_kb

    Returns:
        Merged dictionary with manual entries taking precedence
    """
    merged = dict(auto)
    for term, expansions in manual.items():
        merged[term] = expansions
    return merged
