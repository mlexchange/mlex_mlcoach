from dash import html, dcc
import dash_bootstrap_components as dbc


def sidebar(file_explorer, models, counters):
    '''
    Creates the dash components in the left sidebar of the app
    Args:
        file_explorer:      Dash file explorer
        models:             Currently available ML algorithms in content registry
        counters:           Init training and testing model counters to be used by default when no
                            job description/name is added
    '''
    sidebar = [
        dbc.Card(
            id="sidebar",
            children=[
                dbc.CardHeader("Exploring Data with Machine Learning"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Label('Data'),
                        file_explorer
                    ]),
                    dbc.Button(
                        'Load Labels from Splash-ML', 
                        id='button-load-splash',
                        color='primary', 
                        style={
                            'width': '100%',
                            'margin-top': '10px'
                            }
                        ),
                    dbc.Modal([
                        dbc.ModalHeader(
                            dbc.ModalTitle("Labeling versions")
                            ),
                        dbc.ModalBody(
                            dcc.Dropdown(id='event-id')
                            ),
                        dbc.ModalFooter([
                                dbc.Button(
                                    "LOAD", 
                                    id="confirm-load-splash", 
                                    color='primary', 
                                    outline=False,
                                    className="ms-auto", 
                                    n_clicks=0
                                    )
                                ]
                            ),
                        ], 
                        id="modal-load-splash",
                        is_open=False),
                    dbc.Row([
                        dbc.Label('Action',
                                  style={'margin-top': '10px'}),
                        dcc.Dropdown(
                            id='action',
                            options=[
                                {'label': 'Model Training', 'value': 'train_model'},
                                {'label': 'Test Prediction using Model', 'value': 'prediction_model'},
                            ],
                            value='train_model')
                    ]),
                    dbc.Row([
                        dbc.Label(
                            'Model',
                            style={'margin-top': '10px'}
                            ),
                        dcc.Dropdown(
                            id='model-selection',
                            options=models,
                            value=models[0]['value'])
                    ]),
                    dbc.Button(
                        'Execute',
                        id='execute',
                        n_clicks=0,
                        style={
                            'width': '100%',
                            'margin-left': '0px',
                            'margin-top': '10px'
                            }
                        )
                ])
            ]
        ),
        dbc.Card(
            children=[
                dbc.CardHeader("Parameters"),
                dbc.CardBody([
                    html.Div(id='app-parameters')
                    ])
            ]
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Warning"),
                dbc.ModalBody(id="warning-msg"),
                dbc.ModalFooter([
                    dbc.Button(
                        "OK", 
                        id="ok-button", 
                        color='danger', 
                        outline=False,
                        className="ms-auto", 
                        n_clicks=0
                    ),
                ]),
            ],
            id="warning-modal",
            is_open=False,
        ),
        dcc.Store(id='warning-cause', data=''),
        dcc.Store(id='warning-cause-execute', data=''),
        dcc.Store(id='counters', data=counters)
    ]
    return sidebar