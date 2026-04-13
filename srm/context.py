"""
srm/context.py — multi-turn conversational context management.

This module provides a sliding window context buffer that maintains recent
query/response pairs for follow-up questions and conversational continuity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConversationTurn:
    """A single conversational turn (query + response)."""
    query: str
    response: str
    timestamp: float = field(default_factory=lambda: __import__('time').time())


@dataclass
class ConversationContext:
    """
    Sliding window buffer for conversational context.
    
    Maintains the last N query/response pairs to enable follow-up
    questions like "tell me more about that" or "what about X?".
    """
    max_turns: int = 3
    turns: List[ConversationTurn] = field(default_factory=list)
    
    def add_turn(self, query: str, response: str) -> None:
        """Add a new turn to the context buffer."""
        turn = ConversationTurn(query=query, response=response)
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)
    
    def get_context_string(self, max_turns: Optional[int] = None) -> str:
        """
        Get a formatted context string for prepending to queries.
        
        Args:
            max_turns: Maximum number of recent turns to include (defaults to self.max_turns)
        
        Returns:
            Formatted string with recent conversation history
        """
        limit = max_turns if max_turns is not None else self.max_turns
        recent = self.turns[-limit:] if self.turns else []
        
        if not recent:
            return ""
        
        parts = []
        for turn in recent:
            parts.append(f"Q: {turn.query}")
            parts.append(f"A: {turn.response}")
        
        return " ".join(parts)
    
    def clear(self) -> None:
        """Clear all conversation history."""
        self.turns.clear()
    
    def is_empty(self) -> bool:
        """Check if the context buffer is empty."""
        return len(self.turns) == 0
    
    def last_query(self) -> Optional[str]:
        """Get the most recent query, or None if empty."""
        return self.turns[-1].query if self.turns else None
    
    def last_response(self) -> Optional[str]:
        """Get the most recent response, or None if empty."""
        return self.turns[-1].response if self.turns else None


def augment_query_with_context(
    query: str,
    context: ConversationContext,
    max_context_turns: int = 2,
) -> str:
    """
    Augment a query with relevant context from conversation history.
    
    For follow-up questions (short, ambiguous queries), prepend relevant
    context from recent turns. For substantive queries, leave unchanged.
    
    Args:
        query: The current user query
        context: Conversation context buffer
        max_context_turns: Maximum number of turns to include in augmentation
    
    Returns:
        Augmented query string with context if needed, or original query
    """
    # Skip augmentation for substantive queries (longer than 15 words)
    words = query.split()
    if len(words) > 15:
        return query
    
    # Skip if context is empty
    if context.is_empty():
        return query
    
    # Get recent context
    context_str = context.get_context_string(max_turns=max_context_turns)
    
    # Prepend context to query
    return f"{context_str} Current question: {query}"
