from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
import json
from datetime import datetime

from rag.generator import generate_response

def create_generator_component():
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                DashIconify(icon="mdi:robot-outline", className="me-2"),
                "Rule Generator"
            ], className="mb-0 fw-semibold")
        ]),
        
        dbc.CardBody([
            # Drop zone for dragged rules
            html.Div([
                DashIconify(icon="mdi:cloud-upload-outline", width=32, className="text-muted mb-2"),
                html.P("Drag rules here to analyze", className="text-muted mb-0")
            ], 
            id="drop-zone", 
            className="drop-zone text-center py-4 mb-3"
            ),
            
            # Active rules display
            html.Div(id="active-rules", className="mb-3"),
            
            # Chat interface
            html.Div([
                # Chat messages
                html.Div(
                    id="chat-messages",
                    className="chat-messages mb-3",
                    children=[
                        html.Div([
                            DashIconify(icon="mdi:robot", className="me-2"),
                            html.Span("Hi! I can help you analyze rules, create new ones, or answer questions. Drag some rules here to get started!", 
                                    className="chat-message-text")
                        ], className="chat-message bot-message")
                    ]
                ),
                
                # Input area
                dbc.InputGroup([
                    dbc.Input(
                        id="chat-input",
                        placeholder="Ask me about rules, or request new ones...",
                        className="chat-input"
                    ),
                    dbc.Button([
                        DashIconify(icon="mdi:send", width=20)
                    ], id="send-btn", color="primary")
                ])
            ])
        ])
    ], className="h-100 generator-card")

def create_active_rule_chip(rule):
    return dbc.Badge([
        rule["name"],
        html.Span("×", className="ms-2 remove-rule", **{"data-rule-id": rule.get("id", "")})
    ], color="light", text_color="dark", className="me-2 mb-2 active-rule-chip")

def create_chat_message(content, is_user=True):
    icon = "mdi:account" if is_user else "mdi:robot"
    class_name = "user-message" if is_user else "bot-message"
    
    return html.Div([
        DashIconify(icon=icon, className="me-2"),
        html.Span(content, className="chat-message-text")
    ], className=f"chat-message {class_name}")

def register_generator_callbacks(app):
    @app.callback(
        Output("chat-messages", "children"),
        [Input("send-btn", "n_clicks"),
         Input("chat-input", "n_submit")],
        [State("chat-input", "value"),
         State("chat-messages", "children"),
         State("active-rules", "children")]
    )
    def handle_chat_message(n_clicks, n_submit, message, current_messages, active_rules):
        if not message or not message.strip():
            return current_messages
        
        # Add user message
        new_messages = current_messages + [create_chat_message(message, is_user=True)]
        
        # Generate bot response
        active_rule_data = []  # Extract from active_rules in real implementation
        bot_response = generate_response(message, active_rule_data)
        new_messages.append(create_chat_message(bot_response, is_user=False))
        
        return new_messages
    
    @app.callback(
        Output("chat-input", "value"),
        [Input("send-btn", "n_clicks"),
         Input("chat-input", "n_submit")],
        [State("chat-input", "value")]
    )
    def clear_input(n_clicks, n_submit, value):
        if value:
            return ""
        return value

    # Client-side callback for drag and drop functionality
    app.clientside_callback(
        """
        function(id) {
            const dropZone = document.getElementById('drop-zone');
            const activeRulesDiv = document.getElementById('active-rules');
            
            if (!dropZone) return window.dash_clientside.no_update;
            
            // Handle dragover
            dropZone.addEventListener('dragover', function(e) {
                e.preventDefault();
                dropZone.classList.add('drag-over');
            });
            
            // Handle dragleave
            dropZone.addEventListener('dragleave', function(e) {
                dropZone.classList.remove('drag-over');
            });
            
            // Handle drop
            dropZone.addEventListener('drop', function(e) {
                e.preventDefault();
                dropZone.classList.remove('drag-over');
                
                const ruleData = e.dataTransfer.getData('text/plain');
                if (ruleData) {
                    try {
                        const rule = JSON.parse(ruleData);
                        // Add rule chip to active rules
                        const chip = document.createElement('span');
                        chip.className = 'badge bg-light text-dark me-2 mb-2 active-rule-chip';
                        chip.innerHTML = rule.name + ' <span class="ms-2 remove-rule">×</span>';
                        activeRulesDiv.appendChild(chip);
                    } catch (e) {
                        console.error('Error parsing rule data:', e);
                    }
                }
            });
            
            // Make rule cards draggable
            document.addEventListener('dragstart', function(e) {
                if (e.target.classList.contains('rule-card')) {
                    const ruleData = e.target.getAttribute('data-rule');
                    e.dataTransfer.setData('text/plain', ruleData);
                }
            });
            
            return window.dash_clientside.no_update;
        }
        """,
        Output("drop-zone", "data-initialized"),
        Input("drop-zone", "id")
    )