"""
generator.py — Rule Generator UI

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
from typing import Any, Dict, List, Optional

from dash import Input, Output, State, dcc, html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

# Local import – function that talks to your language-model backend
from rag.generator import generate_response


# ----------------------------------------------------------------------------
# UI FACTORIES
# ----------------------------------------------------------------------------

def create_generator_component() -> dbc.Card:
    """Return the fully-assembled Rule Generator card."""

    return dbc.Card(
        [
            # ---------- Header --------------------------------------------------
            dbc.CardHeader(
                html.H4(
                    [DashIconify(icon="mdi:robot-outline", className="me-2"), "Rule Generator"],
                    className="mb-0 fw-semibold",
                )
            ),
            # ---------- Body ----------------------------------------------------
            dbc.CardBody(
                [
                    # Stores / hidden input ----------------------------------
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
                                    dbc.Input(
                                        id="chat-input",
                                        placeholder="Ask me about rules, or request new ones…",
                                        className="chat-input",
                                    ),
                                    dbc.Button(
                                        DashIconify(icon="mdi:send", width=20),
                                        id="send-btn",
                                        color="primary",
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
        ],
        className="h-100 generator-card",
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

def register_generator_callbacks(app):  # type: ignore[arg-type]
    """Attach all Dash callbacks to app. Call this once during startup."""

    # ----------------------- Chat handling -----------------------------------
    @app.callback(
        Output("chat-messages", "children"),
        [Input("send-btn", "n_clicks"), Input("chat-input", "n_submit")],
        [
            State("chat-input", "value"),
            State("chat-messages", "children"),
            State("stored-rules", "data"),
        ],
    )
    def _handle_chat_message(
        _n_clicks: Optional[int],
        _n_submit: Optional[int],
        message: Optional[str],
        messages: List[Any],
        rules: List[Dict[str, Any]],
    ) -> List[Any]:
        """Append user message & model reply to chat history."""
        if not (message and message.strip()):
            return messages

        updated = messages + [_create_chat_message(message, is_user=True)]
        reply = generate_response(message, rules or [])
        updated.append(_create_chat_message(reply, is_user=False))
        return updated

    # ----------------------- Clear input after send --------------------------
    @app.callback(
        Output("chat-input", "value"),
        [Input("send-btn", "n_clicks"), Input("chat-input", "n_submit")],
        State("chat-input", "value"),
    )
    def _clear_input(_n_clicks: Optional[int], _n_submit: Optional[int], _value: Optional[str]):
        return ""

    # ----------------------- Render active rule chips ------------------------
    @app.callback(Output("active-rules", "children"), Input("stored-rules", "data"))
    def _render_active_rule_chips(rules: List[Dict[str, Any]]):
        return [_create_active_rule_chip(r) for r in rules]

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

        # -- Remove
        if isinstance(action, dict) and "removeId" in action:
            print(f"[REMOVE] Rule ID: {action['removeId']}")
            return [r for r in rules if r.get("rule_id") != action["removeId"]]

        # -- Add
        if isinstance(action, dict) and "rule_id" in action:
            rule_id = action["rule_id"]
            if not any(r.get("rule_id") == rule_id for r in rules):
                print(f"[ADD] Rule object:\n{json.dumps(action, indent=2)}")
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
            if (window._ruleGeneratorInitDone) return true;  // already wired
            window._ruleGeneratorInitDone = true;

            /* Drag-over styling */
            dropZone.addEventListener('dragover', e => {
                e.preventDefault();
                dropZone.classList.add('drag-over');
            });
            dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

            /* Drop handler */
            dropZone.addEventListener('drop', e => {
                e.preventDefault();
                dropZone.classList.remove('drag-over');
                const txt = e.dataTransfer.getData('text/plain');
                if (!txt) return;
                try {
                    const obj = JSON.parse(txt);
                    hiddenInpt.setAttribute('value', JSON.stringify(obj));  // attribute -> triggers Dash
                    hiddenInpt.dispatchEvent(new Event('input', { bubbles: true }));
                } catch (_) {}
            });

            /* Remove rule chip */
            activeDiv.addEventListener('click', e => {
                const removeBtn = e.target.closest('.remove-rule');
                if (!removeBtn) return;
                const id = e.target.getAttribute('data-rule-id');
                hiddenInpt.setAttribute('value', JSON.stringify({ removeId: id }));
                hiddenInpt.dispatchEvent(new Event('input', { bubbles: true }));
            });

            /* Make external rule cards draggable */
            document.addEventListener('dragstart', e => {
                if (!e.target.classList.contains('rule-card')) return;
                const data = e.target.getAttribute('data-rule');
                if (data) {
                    e.dataTransfer.setData('text/plain', data);
                }
            });

            return true;
        }
        """,
        Output("drop-zone-initialized", "data"),
        Input("drop-zone", "id")
    )

__all__ = ["create_generator_component", "register_generator_callbacks"]
