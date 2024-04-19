import dash_bootstrap_components as dbc
from dash import dcc


def sidebar(file_explorer, models, counters):
    """
    Creates the dash components in the left sidebar of the app
    Args:
        file_explorer:      Dash file explorer
        models:             Currently available ML algorithms in content registry
        counters:           Init training and testing model counters to be used by default when no
                            job description/name is added
    """
    sidebar = [
        dbc.Accordion(
            id="sidebar",
            children=[
                dbc.AccordionItem(
                    title="Data selection",
                    children=[
                        file_explorer,
                        dbc.Button(
                            "Load Labels from Splash-ML",
                            id="button-load-splash",
                            color="primary",
                            style={"width": "100%", "margin-top": "10px"},
                        ),
                        dbc.Modal(
                            [
                                dbc.ModalHeader(dbc.ModalTitle("Labeling versions")),
                                dbc.ModalBody(
                                    [
                                        dcc.Input(
                                            id="timezone-browser",
                                            style={"display": "none"},
                                        ),
                                        dcc.Dropdown(id="event-id"),
                                    ]
                                ),
                                dbc.ModalFooter(
                                    [
                                        dbc.Button(
                                            "LOAD",
                                            id="confirm-load-splash",
                                            color="primary",
                                            outline=False,
                                            className="ms-auto",
                                            n_clicks=0,
                                        )
                                    ]
                                ),
                            ],
                            id="modal-load-splash",
                            is_open=False,
                        ),
                    ],
                ),
                dbc.AccordionItem(
                    title="Model configuration",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Label(
                                        "Action",
                                        style={
                                            "height": "100%",
                                            "display": "flex",
                                            "align-items": "center",
                                        },
                                    ),
                                    width=2,
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="action",
                                        options=[
                                            {"label": "Train", "value": "train_model"},
                                            {
                                                "label": "Prediction",
                                                "value": "prediction_model",
                                            },
                                        ],
                                        value="train_model",
                                    ),
                                    width=10,
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Label(
                                        "Model",
                                        style={
                                            "height": "100%",
                                            "display": "flex",
                                            "align-items": "center",
                                        },
                                    ),
                                    width=2,
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="model-selection",
                                        options=models,
                                        value=models[0]["value"],
                                    ),
                                    width=10,
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    id="app-parameters",
                                    style={
                                        "overflowY": "scroll",
                                        "height": "58vh",  # Adjust as needed
                                    },
                                ),
                            ]
                        ),
                        dbc.Button(
                            "Execute",
                            id="execute",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "margin-left": "0px",
                                "margin-top": "10px",
                            },
                        ),
                    ],
                ),
            ],
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Warning"),
                dbc.ModalBody(id="warning-msg"),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "OK",
                            id="ok-button",
                            color="danger",
                            outline=False,
                            className="ms-auto",
                            n_clicks=0,
                        ),
                    ]
                ),
            ],
            id="warning-modal",
            is_open=False,
        ),
        dcc.Store(id="warning-cause", data=""),
        dcc.Store(id="warning-cause-execute", data=""),
        dcc.Store(id="counters", data=counters),
    ]
    return sidebar
