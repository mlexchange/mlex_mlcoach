from dash import html, dcc
import dash_bootstrap_components as dbc

from utils.plot_utils import plot_figure


def main_display(loss_plot, job_table):
    '''
    Creates the dash components within the main display in the app
    Args:
        loss_plot:          Loss plot of trainin process
        job_table:          Job table
    '''
    main_display = html.Div([
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    id="inter_graph",
                    style={"width" : "100%"},
                    children=[
                        dbc.CardHeader('Data Overview'),
                        dbc.CardBody(
                            children=[
                                html.Img(
                                    id='img-output',
                                    src=plot_figure(),
                                    style={'width':'100%'}
                                    ),
                                html.Div(
                                    [
                                        dbc.Label(
                                            id='img-label',
                                            style={'height': '3rem'}),
                                        dcc.Slider(
                                            id='img-slider',
                                            min=0,
                                            step=1,
                                            marks=None,
                                            value=0,
                                            tooltip={
                                                "placement": "bottom",
                                                "always_visible": True
                                                }
                                            ),
                                        dbc.Label('List of labeled images:'),
                                        dcc.Dropdown(
                                            id='img-labeled-indx',
                                            options=[]
                                            ),
                                    ],
                                    style={'vertical-align': 'bottom'}
                                    )
                                ],
                            style={'height': '25rem', 'vertical-align': 'bottom'}
                            ),
                        ]
                    )
                ),
            dbc.Col(
                dbc.Card(
                    id = 'results',
                    children=[
                        dbc.CardHeader('Results'),
                        dbc.CardBody(
                            children = [
                                dcc.Graph(
                                    id='results-plot',
                                    style={'display': 'none'})
                            ],
                            style={'height': '25rem'}
                            )
                        ]
                    ),
                width=7
                )
            ]
        ),
        html.Div(loss_plot),
        job_table,
        dcc.Interval(id='interval', interval=5 * 1000, n_intervals=0),
    ])
    return main_display