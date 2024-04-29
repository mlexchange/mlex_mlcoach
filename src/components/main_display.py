import dash_bootstrap_components as dbc
from dash import dcc, html

from src.utils.plot_utils import plot_figure


def main_display(job_table):
    """
    Creates the dash components within the main display in the app
    Args:
        job_table:          Job table
    """
    main_display = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            id="inter_graph",
                            style={"width": "100%"},
                            children=[
                                dbc.CardHeader("Data Overview"),
                                dbc.CardBody(
                                    children=[
                                        dcc.Loading(
                                            id="loading-display",
                                            parent_className="transparent-loader-wrapper",
                                            children=[
                                                html.Div(
                                                    html.Img(
                                                        id="img-output",
                                                        src=plot_figure(),
                                                        style={
                                                            "height": "60%",
                                                            "display": "block",
                                                            "margin": "auto",
                                                        },
                                                    ),
                                                )
                                            ],
                                        ),
                                        dcc.Store(
                                            id="img-output-store",
                                            data=None,
                                        ),
                                        dcc.Store(id="img-uri", data=None),
                                        html.Div(
                                            [
                                                dbc.Label(
                                                    id="img-label",
                                                    style={"height": "2rem"},
                                                ),
                                                dcc.Slider(
                                                    id="img-slider",
                                                    min=0,
                                                    step=1,
                                                    marks=None,
                                                    value=0,
                                                    tooltip={
                                                        "placement": "bottom",
                                                        "always_visible": True,
                                                    },
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            dbc.Label(
                                                                "List of labeled images:",
                                                                style={
                                                                    "height": "100%",
                                                                    "display": "flex",
                                                                    "align-items": "center",
                                                                },
                                                            ),
                                                        ),
                                                        dbc.Col(
                                                            dcc.Dropdown(
                                                                id="img-labeled-indx",
                                                                options=[],
                                                                clearable=False,
                                                            ),
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            style={"vertical-align": "bottom"},
                                        ),
                                    ],
                                    style={
                                        "height": "45vh",
                                        "vertical-align": "bottom",
                                    },
                                ),
                            ],
                        ),
                        width=5,
                    ),
                    dbc.Col(
                        dbc.Card(
                            id="results",
                            children=[
                                dbc.CardHeader("Results"),
                                dbc.CardBody(
                                    children=[
                                        dcc.Graph(
                                            id="results-plot", style={"display": "none"}
                                        )
                                    ],
                                    style={"height": "45vh"},
                                ),
                            ],
                        ),
                        width=7,
                    ),
                ],
            ),
            job_table,
            dcc.Interval(id="interval", interval=5 * 1000, n_intervals=0),
        ]
    )
    return main_display
