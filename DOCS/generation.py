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
from typing import Any, Dict, List, Optional

from dash import Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

# Local import – function that talks to your language-model backend (stubbed)
from rag.generator import generate_response

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# UI FACTORIES
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
                    dcc.Store(id="session-id"),  # set via clientside UUID on load
                    dcc.Store(id="stored-rules", data=[]),
                    dcc.Input(id="dropped-rule", type="text", style={"display": "none"}),
                    dcc.Store(id="drop-zone-initialized", data=False),

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

                    # Chat interface -----------------------------------------
                    html.Div(
                        [
                            html.Div(
                                id="chat-messages",
                                className="chat-messages mb-3",
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
                                    # IMPORTANT: use dcc.Input so Enter/return triggers n_submit
                                    dcc.Input(
                                        id="chat-input",
                                        placeholder="Ask me about rules, or request new ones…",
                                        type="text",
                                        debounce=False,
                                        className="chat-input form-control",  # form-control to keep styling inside InputGroup
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
            html.Span("×", className="ms-2 remove-rule", **{"data-rule-id": rule_id}),
        ],
        color="light",
        text_color="dark",
        className="me-2 mb-2 active-rule-chip",
    )

def _create_chat_message(content: str, *, is_user: bool = True) -> html.Div:
    """Return a styled chat bubble for user or bot messages."""
    icon_name = "mdi:account" if is_user else "mdi:robot"
    class_name = "user-message" if is_user else "bot-message"
    return html.Div(
        [DashIconify(icon=icon_name, className="me-2"), html.Span(content, className="chat-message-text")],
        className=f"chat-message {class_name}",
    )

# ----------------------------------------------------------------------------
# Callback registration (public)
# ----------------------------------------------------------------------------

def register_chatbot_callbacks(app):  # type: ignore[arg-type]
    """Attach all Dash callbacks to app. Call this once during startup."""

    # ----------------------- Assign per-tab UUID session ---------------------
    app.clientside_callback(
        """
        function(_) {
            if (window._sessionId) return window._sessionId;
            const s4 = () => Math.floor((1 + Math.random()) * 0x10000).toString(16).substring(1);
            const uuid = () => `${s4()}${s4()}-${s4()}-${s4()}-${s4()}-${s4()}${s4()}${s4()}`;
            window._sessionId = uuid();
            return window._sessionId;
        }
        """,
        Output("session-id", "data"),
        Input("drop-zone", "id"),
    )

    # ----------------------- Chat handling -----------------------------------
    @app.callback(
        Output("chat-messages", "children"),
        [Input("send-btn", "n_clicks"), Input("chat-input", "n_submit")],  # Enter or button
        [
            State("chat-input", "value"),
            State("chat-messages", "children"),
            State("stored-rules", "data"),
            State("session-id", "data"),
        ],
        prevent_initial_call=True,
    )
    def _handle_chat_message(
        _n_clicks: Optional[int],
        _n_submit: Optional[int],
        message: Optional[str],
        messages: List[Any],
        rules: List[Dict[str, Any]],
        session_id: Optional[str],
    ) -> List[Any]:
        """Append user message & model reply to chat history (session-scoped)."""
        logger.info("chat: trigger received (send=%s, submit=%s)", _n_clicks, _n_submit)

        # Precondition checks
        if not session_id:
            logger.warning("chat: missing session_id; ignoring turn")
            return messages or []
        if not (message and str(message).strip()):
            logger.debug("chat: empty message; ignoring")
            return messages or []

        updated = (messages or []) + [_create_chat_message(message, is_user=True)]

        try:
            reply = generate_response(message, rules or [], session_id=session_id)
        except Exception as e:
            logger.exception("chat: generate_response failed: %s", e)
            reply = "Sorry, something went wrong while generating a response."

        updated.append(_create_chat_message(reply, is_user=False))
        return updated

    # ----------------------- Clear input after send --------------------------
    @app.callback(
        Output("chat-input", "value"),
        [Input("send-btn", "n_clicks"), Input("chat-input", "n_submit")],
        State("chat-input", "value"),
        prevent_initial_call=True,
    )
    def _clear_input(_n_clicks: Optional[int], _n_submit: Optional[int], _value: Optional[str]):
        return ""

    # ----------------------- Render active rule chips ------------------------
    @app.callback(Output("active-rules", "children"), Input("stored-rules", "data"))
    def _render_active_rule_chips(rules: List[Dict[str, Any]]):
        return [_create_active_rule_chip(r) for r in (rules or [])]

    # ----------------------- Update rule store -------------------------------
    @app.callback(
        Output("stored-rules", "data"),
        Input("dropped-rule", "value"),
        State("stored-rules", "data"),
        prevent_initial_call=True,
    )
    def _update_stored_rules(raw_input: Optional[str], rules: List[Dict[str, Any]]):
        if raw_input is None:
            raise PreventUpdate
        try:
            action = json.loads(raw_input)
        except json.JSONDecodeError:
            raise PreventUpdate

        # Remove
        if isinstance(action, dict) and "removeId" in action:
            rid = action["removeId"]
            return [r for r in (rules or []) if r.get("rule_id") != rid]

        # Add
        if isinstance(action, dict) and "rule_id" in action:
            rule_id = action["rule_id"]
            rules = rules or []
            if not any(r.get("rule_id") == rule_id for r in rules):
                return rules + [action]
            return rules

        raise PreventUpdate

    # ----------------------- One-time JS initialisation ----------------------
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
                const removeBtn = e.target.closest('.remove-rule');
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
