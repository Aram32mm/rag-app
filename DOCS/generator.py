"""
generator.py — Model-backed rule chat (skeleton)

Purpose
-------
Provide a thin, swappable interface used by the Generator UI to produce
chat responses and (later) rule artifacts. Includes a configurable
conversation window and context formatting for selected rules.

Key Responsibilities
--------------------
- Maintain per-session in-memory conversation histories.
- Format active rule context for the model.
- Expose a stable generate_response(user_message, rules, session_id) API.
- Provide stubs for create/modify/explain/suggest functions.

Dependencies
------------
- config constants: GENERATOR_MESSAGE_WINDOW, GENERATOR_TEMPERATURE,
  GENERATOR_MAX_TOKENS, GENERATOR_SYSTEM_PROMPT
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any

from config import (
    GENERATOR_MESSAGE_WINDOW,
    GENERATOR_TEMPERATURE,
    GENERATOR_MAX_TOKENS,
    GENERATOR_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------
# In-memory conversation state (per-session)
# ---------------------------------------
# histories[session_id] = [{"role": "...", "content": "..."}]
_histories: Dict[str, List[Dict[str, str]]] = {}

def _ensure_session(session_id: str) -> None:
    if session_id not in _histories:
        _histories[session_id] = [{"role": "system", "content": GENERATOR_SYSTEM_PROMPT}]
        logger.debug(f"Created new session: {session_id}")

def _trim_history_inplace(session_id: str) -> None:
    hist = _histories.get(session_id, [])
    if not hist:
        return
    sys = hist[0:1]  # keep system prompt
    tail = hist[1:]
    limit = max(0, int(GENERATOR_MESSAGE_WINDOW))
    if len(tail) > limit:
        tail = tail[-limit:]
    _histories[session_id] = sys + tail

def reset_conversation(session_id: str) -> None:
    """Clear a session's conversation back to only the system prompt."""
    _histories[session_id] = [{"role": "system", "content": GENERATOR_SYSTEM_PROMPT}]
    logger.info("Conversation reset for session_id=%s", session_id)

def get_history(session_id: str) -> List[Dict[str, str]]:
    """Return a shallow copy of a session's history."""
    return list(_histories.get(session_id, []))

def _format_rule_context(rules: Optional[List[Dict[str, Any]]]) -> str:
    """Convert active rules into a compact textual context block."""
    rules = rules or []
    if not rules:
        return "Context: No active rules."
    parts = []
    for r in rules[:20]:  # cap to avoid runaway prompts
        rid = str(r.get("rule_id", ""))[:12]
        name = r.get("rule_name", "Unnamed")
        typ = ", ".join(r.get("rule_type", []) or [])
        cty = ", ".join(r.get("country", []) or [])
        parts.append(f"- [{rid}] {name}  (type: {typ or '—'}; country: {cty or '—'})")
    more = "" if len(rules) <= 20 else f"\n… +{len(rules)-20} more"
    return "Context: Active rules\n" + "\n".join(parts) + more

# ---------------------------------------------------
# Public API used by the Generator UI (stubbed logic)
# ---------------------------------------------------
def generate_response(
    user_message: str,
    active_rules: Optional[List[Dict[str, Any]]] = None,
    *,
    session_id: Optional[str] = None,
) -> str:
    """
    Produce a chatbot response given the latest user message and the currently
    selected rules. This is a stub: replace the body with calls to your LLM.

    Args:
        user_message: Latest input from the user.
        active_rules: Rules to be considered as context (may be empty).
        session_id: Unique key for the browser tab/session.

    Returns:
        Assistant reply (string).
    """
    logger.info(f"generate_response called with message: {user_message[:50]}...")
    
    if not session_id:
        logger.warning("generate_response called without session_id; using '_default'.")
        session_id = "_default"

    if not isinstance(user_message, str) or not user_message.strip():
        logger.debug("generate_response called with empty/invalid user_message.")
        return "Please enter a message."

    _ensure_session(session_id)
    ctx = _format_rule_context(active_rules)

    # Add user message to history
    _histories[session_id].append({"role": "user", "content": user_message})
    _trim_history_inplace(session_id)

    # -------- STUB IMPLEMENTATION (replace with your model call) -----------
    reply = (
        "This is a placeholder response.\n\n"
        f"{ctx}\n\n"
        f"You said: "{user_message}". "
        "Connect your LLM in 'rag/generator.py' to generate real answers."
    )
    # -----------------------------------------------------------------------

    # Add assistant reply to history
    _histories[session_id].append({"role": "assistant", "content": reply})
    _trim_history_inplace(session_id)

    logger.info("Reply generated (stub). session_id=%s, hist_len=%d",
                session_id, len(_histories.get(session_id, [])))
    return reply

# ------------------------------------------
# Stubs for rule operations (implement later)
# ------------------------------------------
def create_new_rule(rule_specification: str) -> Dict[str, Any]:
    """Create a new rule based on user specification (stub)."""
    logger.debug("create_new_rule spec=%r", rule_specification)
    return {
        "name": "Generated Rule",
        "description": "Auto-generated based on user input",
        "category": "business",
        "priority": "medium",
        "conditions": [],
        "actions": [],
    }

def modify_rule(rule: Dict[str, Any], modifications: str) -> Dict[str, Any]:
    """Modify an existing rule based on user requests (stub)."""
    logger.debug("modify_rule rule_id=%r mods=%r", rule.get("rule_id"), modifications)
    out = dict(rule)
    out["notes"] = f"Requested modifications: {modifications}"
    return out

def explain_rule(rule: Dict[str, Any]) -> str:
    """Generate a natural language explanation of a rule (stub)."""
    logger.debug("explain_rule rule_id=%r", rule.get("rule_id"))
    name = rule.get("rule_name", "Unnamed")
    desc = rule.get("rule_description", "No description provided.")
    return f"Rule "{name}": {desc}"

def suggest_improvements(rules: List[Dict[str, Any]]) -> List[str]:
    """Suggest improvements for a set of rules (stub)."""
    logger.debug("suggest_improvements count=%d", len(rules))
    if not rules:
        return ["Add at least one rule to receive improvement suggestions."]
    return [
        "Clarify rule scope and inputs.",
        "Add explicit error handling paths.",
        "Include examples and counterexamples in documentation.",
    ]