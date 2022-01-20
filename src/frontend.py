import pathlib
import uuid

import dash
import json
import os
import subprocess

import config as cfg
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_table
import numpy as np
import pandas as pd
import PIL.Image as Image
import plotly.express as px
import plotly.graph_objects as go
import uuid

from helpers import SimpleJob
from helpers import generate_dash_widget, data_processing, create_model, \
    save_model, get_job, generate_figure, get_class_prob
import templates

external_stylesheets = [dbc.themes.BOOTSTRAP, "../assets/segmentation-style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

cf = cfg.Config('src/main.cfg')
TRAIN_DIR = cf['TRAIN_DATA_DIR']
VAL_DIR = cf['VALIDATION_DATA_DIR']
TEST_DIR = cf['TEST_DATA_DIR']
MODEL_DIR = cf['MODEL_SAVE_DIR']
DATA_DIR = str(os.environ['DATA_DIR'])
# DATA_DIR = "/mnt/c/Users/postdoc/Documents/Database/born"
# DATA_DIR = "data"
USER = 'admin'

CLASSES = [subdir for subdir in sorted(os.listdir(TEST_DIR)) if
           os.path.isdir(os.path.join(TEST_DIR, subdir))]
CLASS_NUM = len(CLASSES)

# Load initial data
TRAIN_DATA, VAL_DATA, TEST_DATA = data_processing([0, [], 1], TRAIN_DIR, VAL_DIR, TEST_DIR)

try:
    image = Image.open(TRAIN_DATA.filepaths[0])
except ValueError as e:
    print(e)
fig = px.imshow(image, color_continuous_scale="gray")

DATA_PREPROCESS_WIDGET = [dcc.Graph(id='img-output',
                                    figure=fig),
                          dcc.Slider(id='img-slider',
                                     min=0,
                                     value=0,
                                     tooltip={'always_visible': True,
                                              'placement': 'bottom'}),
                          ]

# Job Status Display
JOB_STATUS = dbc.Card(
    children=[
            dbc.CardHeader("List of Jobs"),
            dbc.CardBody(
                children=[
                    dash_table.DataTable(
                        id='jobs_table',
                        columns=[
                            {'name': 'Job ID', 'id': 'job_id'},
                            {'name': 'Type', 'id': 'job_type'},
                            {'name': 'Status', 'id': 'status'},
                            {'name': 'Experiment ID', 'id': 'experiment_id'},
                            {'name': 'Logs', 'id': 'job_logs'}
                        ],
                        data=[],
                        hidden_columns=['job_id', 'experiment_id'],
                        row_selectable='single',
                        style_cell={'padding': '1rem', 'textAlign': 'left'},
                        fixed_rows={'headers': True},
                        css=[{"selector": ".show-hide", "rule": "display: none"}],
                        style_data_conditional=[
                            {'if': {'column_id': 'status', 'filter_query': '{status} = completed'},
                             'backgroundColor': 'green',
                             'color': 'white'},
                            {'if': {'column_id': 'status', 'filter_query': '{status} = failed'},
                             'backgroundColor': 'red',
                             'color': 'white'}
                        ],
                        style_table={'height': '18rem', 'overflowY': 'auto', 'overflowX': 'scroll'}
                    )
                ],
            )
        ]
    )

#Sidebar content, this includes the titles with the splash-ml entry and query
# button
SIDEBAR = [
    dbc.Card(
        id="sidebar",
        children=[
            dbc.CardHeader("Select an Action"),
            dbc.CardBody([
                dcc.Dropdown(
                    id='action',
                    options=[
                        {'label': 'Model Training', 'value': 'train_model'},
                        {'label': 'Evaluate Model on Data', 'value': 'evaluate_model'},
                        {'label': 'Test Prediction using Model', 'value': 'prediction_model'},
                        # {'label': 'Test Prediction using Model (slider)',
                        #  'value': 'prediction_slider'},
                        {'label': 'Transfer Learning', 'value': 'transfer_learning'},
                        {'label': 'View Images in Categories', 'value': 'view_images'},
                        {'label': 'Save and Load Models', 'value': 'save_load'}],
                    value='train_model')
                ])
        ]
    ),
    dbc.Card(
        children=[
            dbc.CardHeader("Parameters"),
            dbc.CardBody(html.Div(id='app_parameters'))
        ]
    )
]

LOSS_PLOT = dbc.Collapse(id='show-plot',
                         children=dbc.Card(
                             id="plot-card",
                             children=[
                                 dbc.CardHeader("Loss Plot"),
                                 dbc.CardBody([
                                     dcc.Graph(id='loss-plot',
                                               style={'width': '100%', 'height': '20rem'})
                                 ])
                             ]
                         ))

# Content section to the right of the sidebar.  This includes the upload bar
# and graphs to be tagged once loaded into app.
CONTENT = [dbc.Card(
    children=[
        html.Div(id='app_content'),
        html.Button('Execute',
                    id='execute',
                    n_clicks=0,
                    className='m-1'),
        html.Div(id='results'),
        dcc.Interval(id='interval', interval=5 * 1000, n_intervals=0)
    ]),
    LOSS_PLOT,
    JOB_STATUS
]

# Setting up initial webpage layout
app.layout = html.Div([templates.header(),
                       dbc.Container(
                           dbc.Row([dbc.Col(SIDEBAR, width=4),
                                    dbc.Col(CONTENT,
                                            width=8,
                                            style={'align-items': 'center', 'justify-content': 'center'}),
                                    html.Div(id='dummy-output')
                                   ])
                       )])


@app.callback(
    Output('jobs_table', 'data'),
    Output('show-plot', 'is_open'),
    Output('loss-plot', 'figure'),
    Output('results', 'children'),
    Input('interval', 'n_intervals'),
    Input('jobs_table', 'selected_rows'),
    Input('img-slider', 'value'),
    prevent_initial_call=True
)
def update_table(n, row, slider_value):
    job_list = get_job(USER, 'mlcoach')
    data_table = []
    if job_list is not None:
        for job in job_list:
            data_table.insert(0,
                              dict(
                                  job_id=job['uid'],
                                  job_type=job['job_type'],
                                  status=job['status'],
                                  experiment_id=job['container_kwargs']['experiment_id'],
                                  job_logs=job['container_logs'])
                              )
    element = []
    show = False
    fig = go.Figure(go.Scatter(x=[], y=[]))
    if row:
        log = data_table[row[0]]["job_logs"]
        if log:
            if data_table[row[0]]['job_type'] == 'train_model':
                start = log.find('epoch')
                if start > -1 and len(log) > start + 5:
                    fig = generate_figure(log, start)
                    show = True
            if data_table[row[0]]['job_type'] == 'evaluate_model':
                element = dcc.Textarea(value=log,
                                       style={'width': '100%'},
                                       className='mb-2')
            if data_table[row[0]]['job_type'] == 'prediction_model':
                start = log.find('class probability')
                if start > -1 and len(log) > start + 10:
                    text = get_class_prob(log, start, slider_value, CLASSES)
                else:
                    text = ''
                element = dcc.Textarea(value=text,
                                       style={'width': '100%'},
                                       className='mb-2')
    return data_table, show, fig, element


@app.callback(
    Output('app_parameters', 'children'),
    Output('app_content', 'children'),
    Input('action', 'value'),
    prevent_intial_call=True)
def load_parameters_and_content(action_selection):
    # get json file from local files -> change to model registry
    with open('models/' + str(action_selection) + '.json') as f:
        dash_schema = json.load(f)
    parameters = generate_dash_widget(dash_schema)
    if action_selection == 'train_model':
        contents = DATA_PREPROCESS_WIDGET.copy()
    if action_selection == 'evaluate_model':
        contents = DATA_PREPROCESS_WIDGET.copy()
    if action_selection == 'prediction_model':
        contents = DATA_PREPROCESS_WIDGET.copy()
    return parameters.children, html.Div(contents)


# Callback for train model
@app.callback(
    [Output('img-output', 'figure'),
     Output('img-slider', 'max')],
    Input('img-slider', 'value'),
    State('action', 'value'),
    prevent_intial_call=True
)
def refresh_image(img_ind, action_selection):
    try:
        if action_selection == 'train_model':
            image = Image.open(TRAIN_DATA.filepaths[img_ind])
            slider_max = len(TRAIN_DATA)-1
        else:
            image = Image.open(TEST_DATA.filepaths[img_ind])
            slider_max = len(TEST_DATA)-1
    except Exception as e:
        print(e)
    fig = px.imshow(image, color_continuous_scale="gray")
    return fig, slider_max


@app.callback(
    Output('dummy-output', 'children'),
    Input('execute', 'n_clicks'),
    [State({'type': 'labelmaker', 'name': ALL, 'layer': 'input'}, 'value'),
     State('action', 'value'),
     State('jobs_table', 'data'),
     State('jobs_table', 'selected_rows')],
    prevent_intial_call=True)
def execute(clicks, values, action_selection, job_data, row):
    if clicks > 0:
        contents = []
        experiment_id = str(uuid.uuid4())
        out_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER, experiment_id))
        out_path.mkdir(parents=True, exist_ok=True)
        if action_selection == 'train_model':
            data_aug_dict = {'rotation_angle': values[0],
                             'image_flip': values[1],
                             'batch_size': values[2]}
            json_dict = {'data_augmentation': data_aug_dict,
                          'pooling': values[3],
                          'stepoch': values[4],
                          'epochs': values[5],
                          'nn_model': values[6]}
            command = "python3 src/train_model.py"
            directories = [str(out_path)]
        if action_selection == 'evaluate_model':
            training_exp_id = job_data[row[0]]['experiment_id']
            in_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER, training_exp_id))
            json_dict = {'rotation_angle': values[0],
                         'image_flip': values[1],
                         'batch_size': values[2]}
            command = "python3 src/evaluate_model.py"
            directories = [str(in_path), str(out_path)]
        if action_selection == 'prediction_model':
            training_exp_id = job_data[row[0]]['experiment_id']
            in_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER, training_exp_id))
            json_dict = {'rotation_angle': values[0],
                         'image_flip': values[1],
                         'batch_size': values[2]}
            command = "python3 src/predict_model.py"
            directories = [str(in_path), str(out_path)]
        job = SimpleJob(user=USER,
                        job_type=action_selection,
                        description='',
                        deploy_location='local',
                        gpu=False,
                        data_uri='{}'.format(DATA_DIR),
                        container_uri='mlexchange/labelmaker-functions',
                        container_cmd=command,
                        container_kwargs={'parameters': json_dict,
                                          'directories': directories,
                                          'experiment_id': experiment_id}
                        )
        job.launch_job()
        return contents
    return []


@app.callback(
    Output('training_loss', 'figure'),
    Input('interval', 'n_intervals'),
    prevent_initial_call=True
)
def plot_loss(n):
    job = get_job(USER, 'mlcoach')
    print(job)
    job = job[0]
    logs = job['container_logs']
    if logs:
        df = pd.read_csv(StringIO(logs), sep=' ')
        fig = px.line(df, x="epoch", y="loss")
        fig.update_traces(mode='markers+lines')
        return fig


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')#, dev_tools_ui=False)
