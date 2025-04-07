import dash_bootstrap_components as dbc
from dash import dcc, html
from mlex_utils.dash_utils.components_bootstrap.component_utils import (
    DbcControlItem as ControlItem,
)

from src.utils.plot_utils import plot_figure


def main_display(loss_plot):
    """
    Creates the dash components within the main display in the app
    Args:
        loss_plot:      Loss plot
    """
    main_display = html.Div(
        id="main-display",
        style={"padding": "0px 10px 0px 510px"},
        children=[
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
                                                    [
                                                        html.Img(
                                                            id="img-output",
                                                            src=plot_figure(),
                                                            style={"height": "60%"},
                                                        ),
                                                        dcc.Store(
                                                            id="img-output-store",
                                                            data=None,
                                                        ),
                                                    ],
                                                    style={
                                                        "display": "flex",
                                                        "justify-content": "center",
                                                        "width": "100%",
                                                    },
                                                ),
                                            ],
                                        ),
                                        dcc.Store(id="img-uri", data=None),
                                        html.P(),
                                        html.Div(
                                            [
                                                dbc.Label(
                                                    id="img-label",
                                                    style={"height": "2rem"},
                                                ),
                                                html.P(),
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
                                                html.P(),
                                                ControlItem(
                                                    "Labeled images:",
                                                    "title-img-labeled-indx",
                                                    dcc.Loading(
                                                        id="loading-labeled-imgs",
                                                        parent_className="transparent-loader-wrapper",
                                                        children=[
                                                            dbc.Select(
                                                                id="img-labeled-indx",
                                                                options=[],
                                                            ),
                                                        ],
                                                    ),
                                                ),
                                            ],
                                            style={"vertical-align": "bottom"},
                                        ),
                                    ],
                                    style={
                                        "height": "83vh",
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
                                    style={"height": "83vh"},
                                ),
                            ],
                        ),
                        width=7,
                    ),
                ],
            ),
            html.Div(loss_plot),
            dcc.Interval(id="interval", interval=5 * 1000, n_intervals=0),
        ],
    )
    return main_display
