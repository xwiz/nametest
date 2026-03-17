from __future__ import annotations

import re
from dataclasses import dataclass


_GREETING_RE = re.compile(r"^(hi|hello|hey|thanks|thank you|good morning|good afternoon|good evening)\b", re.IGNORECASE)
_DECLARATIVE_RE = re.compile(
    r"^(?:i am|i'm|i feel|i want|i need|i like|i love|i prefer|i have|i think|i believe|i remember|please|could you|can you|will you)\b",
    re.IGNORECASE,
)
_BAD_SECRET_RE = re.compile(r"\b(password|passcode|api key|secret|token|ssn|social security|credit card)\b", re.IGNORECASE)
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_LONG_DIGIT_RE = re.compile(r"\d{6,}")
_CODE_RE = re.compile(r"[`{}<>;$]|\b(class|def|function|SELECT|INSERT|UPDATE|DELETE)\b")
_EMERGENCY_RE = re.compile(r"\b(fire|bleeding|suicid|overdose|can'?t breathe|stroke|heart attack)\b", re.IGNORECASE)
_PII_RE = re.compile(r"\b(my name is|call me|i live at|my address|my phone|my number is)\b", re.IGNORECASE)


REJECTION_DESCRIPTIONS: dict[str, str] = {
    "empty":             "Input was empty.",
    "command":           "Starts with '/' — treated as a REPL command, not learnable text.",
    "too_short":         "Too short (< 3 chars) to be useful as a memory.",
    "too_long":          "Too long (> 180 chars) — likely a paste, not conversational input.",
    "url":               "Contains a URL — URLs are not stored as memories.",
    "email":             "Contains an email address — potential PII.",
    "long_number":       "Contains a long number sequence (≥ 6 digits) — potential PII/code.",
    "secret":            "Contains secret-like keywords (password, token, API key, etc.).",
    "pii":               "Contains personally identifiable information patterns.",
    "code_like":         "Looks like code or a database query — not conversational.",
    "emergency":         "Contains emergency keywords — not safe to auto-store.",
    "not_enough_words":  "Fewer than 2 alphabetic words — not enough content.",
    "question_pattern":  "Ends with '?' — questions are queried, not stored.",
    "pattern_miss":      "Does not match any safe conversational pattern (greeting, declaration, etc.).",
}


@dataclass(frozen=True)
class LearningDecision:
    accepted: bool
    reason: str
    normalized_text: str | None = None

    @property
    def explanation(self) -> str:
        """Human-readable explanation of why this input was accepted or rejected."""
        if self.accepted:
            return f"Accepted ({self.reason}): '{self.normalized_text}'"
        return REJECTION_DESCRIPTIONS.get(self.reason, f"Rejected: {self.reason}")


def normalize_learning_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def should_auto_learn(text: str) -> LearningDecision:
    raw = text.strip()
    if not raw:
        return LearningDecision(False, "empty")
    if raw.startswith("/"):
        return LearningDecision(False, "command")
    if len(raw) < 3:
        return LearningDecision(False, "too_short")
    if len(raw) > 180:
        return LearningDecision(False, "too_long")
    if _URL_RE.search(raw):
        return LearningDecision(False, "url")
    if _EMAIL_RE.search(raw):
        return LearningDecision(False, "email")
    if _LONG_DIGIT_RE.search(raw):
        return LearningDecision(False, "long_number")
    if _BAD_SECRET_RE.search(raw):
        return LearningDecision(False, "secret")
    if _PII_RE.search(raw):
        return LearningDecision(False, "pii")
    if _CODE_RE.search(raw):
        return LearningDecision(False, "code_like")
    if _EMERGENCY_RE.search(raw):
        return LearningDecision(False, "emergency")

    normalized = normalize_learning_text(raw)
    low = normalized.lower()
    alpha_words = re.findall(r"[a-z]+", low)
    if len(alpha_words) < 2:
        return LearningDecision(False, "not_enough_words")

    if _GREETING_RE.match(low):
        return LearningDecision(True, "greeting", normalized)
    if _DECLARATIVE_RE.match(low):
        return LearningDecision(True, "conversation_pattern", normalized)
    if normalized.endswith("?"):
        return LearningDecision(False, "question_pattern")
    return LearningDecision(False, "pattern_miss")


def explain_rejection(text: str) -> str:
    """Return a human-readable explanation of the learning decision for *text*.

    Useful for REPL debug output so users understand why an input
    was or was not auto-learned.
    """
    return should_auto_learn(text).explanation
