"""
chatbot.py — Chatbot Rule Generator UI

Purpose
-------
Dash component for dragging rule definitions into a drop-zone, chatting with a
language-model backend, and generating new rule proposals.

Key Responsibilities
--------------------
- Provide drop-zone for rule cards (drag-and-drop).
- Display active rules as removable chips.
- Maintain chat interface with model responses.
- Manage stored rules (add/remove via JSON actions).
- Initialize client-side JS for drag/drop event handling.

Dependencies
------------
- dash, dash-bootstrap-components
- rag.generator.generate_response
"""

from __future__ import annotations

import json
import logging
from time import time
from typing import Any, Dict, List, Optional, Tuple

from dash import Input, Output, State, dcc, html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

# Model-side stub API (synchronous)
from rag.generator import generate_response

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# UI FACTORY
# ----------------------------------------------------------------------------
def create_chatbot_component() -> dbc.Card:
    """Return the fully-assembled Chatbot card."""

    return dbc.Card(
        [
            # ---------- Header --------------------------------------------------
            dbc.CardHeader(
                html.H4(
                    [DashIconify(icon="mdi:robot-outline", className="me-2"), "Chatbot"],
                    className="mb-0 fw-semibold",
                )
            ),

            # ---------- Body ----------------------------------------------------
            dbc.CardBody(
                [
                    # Session + stores / hidden inputs -------------------------
                    # session-id: per-tab, in-memory only (never persisted) to ensure isolation
                    dcc.Store(id="session-id"),
                    # Persist selected rules across refresh (but not across browser sessions)
                    dcc.Store(id="stored-rules", data=[], storage_type="session"),
                    # Persist chat history by session (we namespace by session-id → see callbacks)
                    dcc.Store(id="chat-history", storage_type="session"),
                    # Drag/drop plumbing
                    dcc.Input(id="dropped-rule", type="text", style={"display": "none"}),
                    dcc.Store(id="drop-zone-initialized", data=False),
                    # Two-phase send plumbing
                    dcc.Store(id="response-trigger", data=None),
                    dcc.Store(id="chat-sending", data=False),  # busy flag to control Send button

                    # Drop-zone ------------------------------------------------
                    html.Div(
                        [
                            DashIconify(icon="mdi:cloud-upload-outline", width=32, className="text-muted mb-2"),
                            html.P("Drag rules here to analyse", className="text-muted mb-0"),
                        ],
                        id="drop-zone",
                        className="drop-zone text-center py-4 mb-3",
                    ),

                    # Active rule chips ---------------------------------------
                    html.Div(id="active-rules", className="mb-3"),

                    # Chat interface ------------------------------------------
                    html.Div(
                        [
                            html.Div(
                                id="chat-messages",
                                className="chat-messages mb-3",
                                # Default welcome message (replaced with history on load if present)
                                children=[
                                    _create_chat_message(
                                        "Hi! I can help you analyse rules, create new ones, or answer questions. "
                                        "Drag some rules here to get started!",
                                        is_user=False,
                                    )
                                ],
                            ),
                            dbc.InputGroup(
                                [
                                    # Use dcc.Input so Enter triggers n_submit
                                    dcc.Input(
                                        id="chat-input",
                                        placeholder="Ask me about rules, or request new ones…",
                                        type="text",
                                        debounce=False,
                                        n_submit=0,
                                        className="chat-input form-control",
                                    ),
                                    dbc.Button(
                                        DashIconify(icon="mdi:send", width=20),
                                        id="send-btn",
                                        color="primary",
                                        n_clicks=0,
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
        ],
        className="h-100 chatbot-card",
    )


# ----------------------------------------------------------------------------
# Helper builders (private)
# ----------------------------------------------------------------------------
def _create_active_rule_chip(rule: Dict[str, Any]) -> dbc.Badge:
    """Return a pill-style badge representing an active rule."""
    rule_id = rule.get("rule_id", "")
    return dbc.Badge(
        [
            rule.get("rule_name", "Unnamed"),
            html.Span("×", className="ms-2 remove-rule remove-rule-btn", **{"data-rule-id": rule_id}),
        ],
        color="light",
        text_color="dark",
        className="me-2 mb-2 active-rule-chip",
    )


def _create_chat_message(content: str, *, is_user: bool = True, is_loading: bool = False) -> html.Div:
    """Return a styled chat bubble for user or bot messages (supports loading spinner)."""
    if is_loading:
        return html.Div(
            [
                DashIconify(icon="mdi:robot", className="me-2"),
                dbc.Spinner(size="sm", spinner_class_name="chat-message-text"),
            ],
            className="chat-message bot-message",
            id="loading-message",
        )

    icon_name = "mdi:account" if is_user else "mdi:robot"
    class_name = "user-message" if is_user else "bot-message"
    return html.Div(
        [
            DashIconify(icon=icon_name, className="me-2"),
            dcc.Markdown(content, className="chat-message-text", link_target="_blank"),
        ],
        className=f"chat-message {class_name}",
    )


# ----------------------------------------------------------------------------
# Callback registration (public)
# ----------------------------------------------------------------------------
def register_chatbot_callbacks(app):  # type: ignore[arg-type]
    """Attach all Dash callbacks to app. Call this once during startup."""

    # ---------- Per-tab session id (in-memory only; never persisted) ----------
    app.clientside_callback(
        """
        function(_) {
            // Try per-tab sessionStorage first (persists across reloads in the same tab)
            try {
                let id = sessionStorage.getItem("chatbot_tab_session_id");
                if (!id) {
                    // Use secure random if available, else fallback to a uuid-ish function
                    if (window.crypto && crypto.randomUUID) {
                        id = crypto.randomUUID();
                    } else {
                        const s4 = () => Math.floor((1 + Math.random()) * 0x10000).toString(16).substring(1);
                        id = `${s4()}${s4()}-${s4()}-${s4()}-${s4()}-${s4()}${s4()}${s4()}`;
                    }
                    sessionStorage.setItem("chatbot_tab_session_id", id);
                }
                return id;
            } catch (e) {
                // If sessionStorage is blocked, fall back to an in-memory id (lost on reload)
                if (!window._tabSessionId) {
                    if (window.crypto && crypto.randomUUID) {
                        window._tabSessionId = crypto.randomUUID();
                    } else {
                        const s4 = () => Math.floor((1 + Math.random()) * 0x10000).toString(16).substring(1);
                        window._tabSessionId = `${s4()}${s4()}-${s4()}-${s4()}-${s4()}-${s4()}${s4()}${s4()}`;
                    }
                }
                return window._tabSessionId;
            }
        }
        """,
        Output("session-id", "data"),
        Input("drop-zone", "id"),
    )

    # ---------- Load chat history (namespaced by session-id) on open ----------
    @app.callback(
        Output("chat-messages", "children", allow_duplicate=True),
        Input("session-id", "data"),
        State("chat-history", "data"),
        prevent_initial_call=True,
    )
    def _load_history_on_open(session_id: Optional[str], history_by_sid: Optional[Dict[str, List[Any]]]):
        if not session_id:
            logger.warning("[Chat UI] No session-id on open; leaving default welcome message.")
            raise PreventUpdate

        history_by_sid = history_by_sid or {}
        history = history_by_sid.get(session_id)

        if history:
            logger.info("[Chat UI] Restored %d chat message(s) for session %s.", len(history), session_id)
            return history

        logger.info("[Chat UI] No prior chat history for session %s; showing default welcome.", session_id)
        # Return default welcome if no history
        return [
            _create_chat_message(
                "Hi! I can help you analyse rules, create new ones, or answer questions. "
                "Drag some rules here to get started!",
                is_user=False,
            )
        ]

    # ---------- Phase 1: accept submission & show loader ----------
    @app.callback(
        Output("chat-messages", "children", allow_duplicate=True),
        Output("response-trigger", "data"),
        Output("chat-sending", "data", allow_duplicate=True),  # set True
        [Input("send-btn", "n_clicks"), Input("chat-input", "n_submit")],
        [
            State("chat-input", "value"),
            State("chat-messages", "children"),
            State("stored-rules", "data"),
            State("session-id", "data"),
            State("chat-sending", "data"),
        ],
        prevent_initial_call=True,
    )
    def _handle_chat_submission(
        _n_clicks: int,
        _n_submit: int,
        message: Optional[str],
        messages: List[Any],
        rules: List[Dict[str, Any]],
        session_id: Optional[str],
        sending: Optional[bool],
    ) -> Tuple[List[Any], Dict[str, Any] | None, bool]:

        if sending:
            logger.debug("[Chat UI] Submission ignored: still sending.")
            raise PreventUpdate
        if not session_id:
            logger.warning("[Chat UI] Submission ignored: missing session_id.")
            raise PreventUpdate
        if not (message and str(message).strip()):
            logger.debug("[Chat UI] Submission ignored: empty message.")
            raise PreventUpdate

        messages = messages or []

        logger.info("[Chat UI] New submission (send=%s, submit=%s) for session %s.", _n_clicks, _n_submit, session_id)
        updated_messages = messages + [
            _create_chat_message(message, is_user=True),
            _create_chat_message("", is_user=False, is_loading=True),
        ]

        trigger_data = {
            "message": message,
            "rules": rules,
            "session_id": session_id,
            "seq": int(time() * 1000),  # force Store delta each turn
        }

        logger.debug("[Chat UI] Emitting response trigger for session %s (seq=%s).", session_id, trigger_data["seq"])
        return updated_messages, trigger_data, True  # now busy

    # ---------- Phase 2: compute reply & replace loader ----------
    @app.callback(
        Output("chat-messages", "children", allow_duplicate=True),
        Output("chat-sending", "data", allow_duplicate=True),  # set False
        Input("response-trigger", "data"),
        State("chat-messages", "children"),
        prevent_initial_call=True,
    )
    def _handle_chat_response(
        trigger_data: Dict[str, Any] | None,
        messages: List[Any],
    ) -> Tuple[List[Any], bool]:

        if not trigger_data:
            logger.debug("[Chat UI] Response phase invoked without trigger_data.")
            raise PreventUpdate

        messages = messages or []

        message = trigger_data["message"]
        rules = trigger_data["rules"]
        session_id = trigger_data["session_id"]
        seq = trigger_data.get("seq")

        logger.info("[Chat UI] Generating response for session %s (seq=%s).", session_id, seq)

        try:
            reply = generate_response(message, rules, session_id=session_id)
        except Exception as e:
            logger.exception("[Chat UI] generate_response failed for session %s: %s", session_id, e)
            reply = "Sorry, something went wrong while generating a response."

        updated_messages = messages[:-1] + [_create_chat_message(reply, is_user=False)]
        logger.info("[Chat UI] Response ready; replacing loader (session=%s, seq=%s).", session_id, seq)

        return updated_messages, False  # no longer busy

    # ---------- Persist chat history (namespaced by session-id) ----------
    @app.callback(
        Output("chat-history", "data"),
        Input("chat-messages", "children"),
        State("session-id", "data"),
        State("chat-history", "data"),
        prevent_initial_call=True,
    )
    def _persist_history(messages: List[Any], session_id: Optional[str], history_by_sid: Optional[Dict[str, List[Any]]]):
        if not session_id:
            logger.warning("[Chat UI] Skipping history persist: missing session_id.")
            raise PreventUpdate

        history_by_sid = history_by_sid or {}
        history_by_sid[session_id] = messages or []
        logger.debug("[Chat UI] Persisted %d message(s) for session %s.", len(history_by_sid[session_id]), session_id)
        return history_by_sid

    # ---------- Map busy store -> disable Send button (single writer) ----------
    @app.callback(
        Output("send-btn", "disabled"),
        Input("chat-sending", "data"),
    )
    def _toggle_send_disabled(sending: Optional[bool]) -> bool:
        disabled = bool(sending)
        logger.debug("[Chat UI] Send button disabled=%s.", disabled)
        return disabled

    # ---------- Clear input after send ----------
    @app.callback(
        Output("chat-input", "value"),
        [Input("send-btn", "n_clicks"), Input("chat-input", "n_submit")],
        State("chat-input", "value"),
        prevent_initial_call=True,
    )
    def _clear_input(_n_clicks: Optional[int], _n_submit: Optional[int], _value: Optional[str]):
        logger.debug("[Chat UI] Clearing chat input after send.")
        return ""

    # ---------- Render active rule chips ----------
    @app.callback(Output("active-rules", "children"), Input("stored-rules", "data"))
    def _render_active_rule_chips(rules: List[Dict[str, Any]]):
        chips = [_create_active_rule_chip(r) for r in (rules or [])]
        logger.debug("[Chat UI] Rendered %d active rule chip(s).", len(chips))
        return chips

    # ---------- Update rule store (add/remove) ----------
    @app.callback(
        Output("stored-rules", "data"),
        Input("dropped-rule", "value"),
        State("stored-rules", "data"),
        prevent_initial_call=True,
    )
    def _update_stored_rules(raw_input: Optional[str], rules: List[Dict[str, Any]]):
        if raw_input is None:
            logger.debug("[Chat UI] No drop payload; ignoring.")
            raise PreventUpdate

        try:
            action = json.loads(raw_input)
        except json.JSONDecodeError:
            logger.warning("[Chat UI] Invalid drop payload (not JSON); ignoring.")
            raise PreventUpdate

        # Remove
        if isinstance(action, dict) and "removeId" in action:
            rid = action["removeId"]
            new_rules = [r for r in (rules or []) if r.get("rule_id") != rid]
            logger.info("[Chat UI] Removed rule_id=%s (chips now=%d).", rid, len(new_rules))
            return new_rules

        # Add
        if isinstance(action, dict) and "rule_id" in action:
            rule_id = action["rule_id"]
            rules = rules or []
            if not any(r.get("rule_id") == rule_id for r in rules):
                logger.info("[Chat UI] Added rule_id=%s (chips now=%d).", rule_id, len(rules) + 1)
                return rules + [action]
            logger.debug("[Chat UI] Drop ignored: rule_id=%s already present.", rule_id)
            return rules

        logger.debug("[Chat UI] Drop payload had no actionable keys; ignoring.")
        raise PreventUpdate

    # ---------- One-time JS for drag/drop + chip removal ----------
    app.clientside_callback(
        """
        function (_) {
            const dropZone   = document.getElementById('drop-zone');
            const activeDiv  = document.getElementById('active-rules');
            const hiddenInpt = document.getElementById('dropped-rule');

            if (!dropZone || !hiddenInpt) return false;
            if (window._ruleGeneratorInitDone) return true;
            window._ruleGeneratorInitDone = true;

            // Drag-over styling
            dropZone.addEventListener('dragover', e => {
                e.preventDefault();
                dropZone.classList.add('drag-over');
            });
            dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

            // Drop handler
            dropZone.addEventListener('drop', e => {
                e.preventDefault();
                dropZone.classList.remove('drag-over');
                const txt = e.dataTransfer.getData('text/plain');
                if (!txt) return;
                try {
                    const obj = JSON.parse(txt);
                    hiddenInpt.setAttribute('value', JSON.stringify(obj));
                    hiddenInpt.dispatchEvent(new Event('input', { bubbles: true }));
                } catch (_) {}
            });

            // Remove rule chip
            activeDiv.addEventListener('click', e => {
                const removeBtn = e.target.closest('.remove-rule-btn');
                if (!removeBtn) return;
                const id = removeBtn.getAttribute('data-rule-id');
                if (!id) return;
                hiddenInpt.setAttribute('value', JSON.stringify({ removeId: id }));
                hiddenInpt.dispatchEvent(new Event('input', { bubbles: true }));
            });

            // Make external rule cards draggable
            document.addEventListener('dragstart', e => {
                if (!e.target.classList.contains('rule-card')) return;
                const data = e.target.getAttribute('data-rule');
                if (data) e.dataTransfer.setData('text/plain', data);
            });

            return true;
        }
        """,
        Output("drop-zone-initialized", "data"),
        Input("drop-zone", "id")
    )


__all__ = ["create_chatbot_component", "register_chatbot_callbacks"]
