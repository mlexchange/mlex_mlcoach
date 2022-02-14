import pathlib
import dash
import os

import config as cfg
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_table
import pandas as pd
import PIL.Image as Image
import plotly.express as px
import plotly.graph_objects as go
import uuid

from helpers import SimpleJob
from helpers import get_job, generate_figure, get_class_prob, model_list_GET_call
from kwarg_editor import JSONParameterEditor
import templates

external_stylesheets = [dbc.themes.BOOTSTRAP, "../assets/segmentation-style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# Path to dataset folders
cf = cfg.Config('src/main.cfg')
TRAIN_DIR = cf['TRAIN_DATA_DIR']
VAL_DIR = cf['VALIDATION_DATA_DIR']
TEST_DIR = cf['TEST_DATA_DIR']
MODEL_DIR = cf['MODEL_SAVE_DIR']

# Global variables
DATA_DIR = str(os.environ['DATA_DIR'])
USER = 'admin'
CLASSES = [subdir for subdir in sorted(os.listdir(TEST_DIR)) if
           os.path.isdir(os.path.join(TEST_DIR, subdir))]
CLASS_NUM = len(CLASSES)

# Get training filenames
path, train_folders, extra_files = next(os.walk(TRAIN_DIR))
list_train_filename = []
for class_folder in train_folders:
    path, list_dirs, filenames = next(os.walk(TRAIN_DIR+'/'+class_folder))
    for filename in filenames:
        if filename.split('.')[-1] in ['tiff', 'tif', 'jpg', 'jpeg', 'png']:
            list_train_filename.append(class_folder+'/'+filename)

# Get testing filenames
path, test_folders, extra_files = next(os.walk(TEST_DIR))
list_test_filename = []
for class_folder in test_folders:
    path, list_dirs, filenames = next(os.walk(TEST_DIR+'/'+class_folder))
    for filename in filenames:
        if filename.split('.')[-1] in ['tiff', 'tif', 'jpg', 'jpeg', 'png']:
            list_test_filename.append(class_folder+'/'+filename)

# Loads first image
try:
    image = Image.open(TRAIN_DIR+'/'+list_train_filename[0])
except ValueError as e:
    print(e)
fig = px.imshow(image, color_continuous_scale="gray")

# Reactive component to display images
DATA_PREPROCESS_WIDGET = [dcc.Graph(id='img-output',
                                    figure=fig),
                          dcc.Slider(id='img-slider',
                                     min=0,
                                     value=0,
                                     tooltip={'always_visible': True,
                                              'placement': 'bottom'}),
                          ]

# Extra parameters for transfer learning
EXTRA_PARAMETERS = [
            dbc.FormGroup([
                dbc.Label('Pooling Options'),
                dbc.RadioItems(
                    options=[{'label': 'None', 'value': 'None'},
                             {'label': 'Maximum', 'value': 'Maximum'},
                             {'label': 'Average', 'value': 'Average'}],
                    value='None'
                )
            ]),
            dbc.FormGroup([
                dbc.Label('Number of epochs'),
                dcc.Slider(id='epochs',
                           min=1,
                           max=1000,
                           tooltip={'always_visible': True,
                                    'placement': 'bottom'})
            ])]

# Data augmentation widget
DATA_AUG_WIDGET = [
    dbc.FormGroup([
        dbc.Label('Rotation Angle'),
        dcc.Slider(id='rotation_angle',
                   min=0,
                   max=360,
                   value=0,
                   tooltip={'always_visible': True,
                            'placement': 'bottom'})
    ]),
    dbc.FormGroup([
        dbc.Label('Image Flip'),
        dbc.RadioItems(
            id='image_flip',
            options=[
               {'label': 'None', 'value': 'None'},
               {'label': 'Vertical', 'value': 'vert'},
               {'label': 'Horizontal', 'value': 'Horizontal'},
               {'label': 'Both', 'value': 'Both'}
            ],
            value = 'None'
        )
    ]),
    dbc.FormGroup([
        dbc.Label('Batch Size'),
        dcc.Slider(id='batch_size',
                  min=16,
                  max=128,
                  value=32,
                  step=16,
                  tooltip={'always_visible': True,
                           'placement': 'bottom'})
    ])
]

# Job Status Display
JOB_STATUS = dbc.Card(
    children=[
            dbc.CardHeader("List of Jobs"),
            dbc.CardBody(
                children=[
                    dash_table.DataTable(
                        id='jobs-table',
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

# Sidebar with actions, model, and parameters selection
SIDEBAR = [
    dbc.Card(
        id="sidebar",
        children=[
            dbc.CardHeader("Select an Action & a Model"),
            dbc.CardBody([
                dbc.FormGroup([
                    dbc.Label('Action'),
                    dcc.Dropdown(
                        id='action',
                        options=[
                            {'label': 'Model Training', 'value': 'train_model'},
                            {'label': 'Evaluate Model on Data', 'value': 'evaluate_model'},
                            {'label': 'Test Prediction using Model', 'value': 'prediction_model'},
                            {'label': 'Transfer Learning', 'value': 'transfer_learning'},
                            # {'label': 'View Images in Categories', 'value': 'view_images'},
                            # {'label': 'Save and Load Models', 'value': 'save_load'}
                        ],
                        value='train_model')
                ]),
                dbc.FormGroup([
                    dbc.Label('Model'),
                    dcc.Dropdown(
                        id='model-selection',
                        options=[
                            {'label': 'Tensorflow Neural Networks', 'value': 'tf-NN'}],
                        value='tf-NN')
                ])
            ])
        ]
    ),
    dbc.Card(
        children=[
            dbc.CardHeader("Parameters"),
            dbc.CardBody(html.Div(id='app-parameters'))
        ]
    )
]

# App contents (right hand side)
CONTENT = [dbc.Card(
    children=[
        dbc.Row([dbc.Col(children = [html.Div(id='app-content'),
                                     dbc.Button('Execute',
                                                id='execute',
                                                n_clicks=0,
                                                className='m-1',
                                                style={'width': '95%', 'justify-content': 'center'})],
                         style={'align-items': 'center', 'justify-content': 'center'}),
                 dbc.Col(html.Div(id='results',
                                  children=dbc.Card(children=[
                                      dbc.CardHeader(id='result-title'),
                                      dbc.CardBody([
                                          dcc.Graph(id='result-plot',
                                                    style={'width': '100%'})    #, 'height': '20rem'
                                      ])]
                                  )
                                  ))
                 ]),
        dcc.Interval(id='interval', interval=5 * 1000, n_intervals=0)
    ]),
    JOB_STATUS
]

# Setting up initial webpage layout
app.layout = html.Div([templates.header(),
                       dbc.Container(
                           dbc.Row([dbc.Col(SIDEBAR, width=3),
                                    dbc.Col(CONTENT,
                                            width=9,
                                            style={'align-items': 'center', 'justify-content': 'center'}),
                                    html.Div(id='dummy-output')
                                   ]),
                           fluid=True
                       )])


@app.callback(
    Output('jobs-table', 'data'),
    Output('result-title', 'children'),
    Output('result-plot', 'figure'),
    Input('interval', 'n_intervals'),
    Input('jobs-table', 'selected_rows'),
    Input('img-slider', 'value'),
    prevent_initial_call=True
)
def update_table(n, row, slider_value):
    '''
    This callback updates the job table, loss plot, and results according to the job status in the compute service.
    Args:
        n:              Time intervals that triggers this callback
        row:            Selected row (job)
        slider_value:   Image slider value (current image)
    Returns:
        jobs-table:     Updates the job table
        show-plot:      Shows/hides the loss plot
        loss-plot:      Updates the loss plot according to the job status (logs)
        results:        Testing results (probability)
    '''
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
    fig = go.Figure(go.Scatter(x=[], y=[]))
    if row:
        log = data_table[row[0]]["job_logs"]
        if log:
            if data_table[row[0]]['job_type'] == 'train_model':
                start = log.find('epoch')
                if start > -1 and len(log) > start + 5:
                    fig = generate_figure(log, start)
                title = 'Loss PLot'
                # element = dbc.Card(children=[
                #                        dbc.CardHeader("Loss Plot"),
                #                        dbc.CardBody([
                #                            dcc.Graph(figure=fig,
                #                                      style={'width': '100%', 'height': '20rem'})
                #                        ])]
                #                    )
            if data_table[row[0]]['job_type'] == 'evaluate_model':
                title = 'Evaluation'
                # element = dcc.Textarea(value=log,
                #                        style={'width': '100%'},
                #                        className='mb-2')
            if data_table[row[0]]['job_type'] == 'prediction_model':
                title = 'Prediction'
                start = log.find('filename')
                if start > -1 and len(log) > start + 10:
                    fig = get_class_prob(log, start, list_test_filename[slider_value])
                    # element = dbc.Card(children=[
                    #                        dbc.CardHeader("Prediction"),
                    #                        dbc.CardBody([
                    #                            dcc.Graph(figure=fig,
                    #                                      style={'width': '100%', 'height': '20rem'})
                    #                        ])]
                    #                    )
                # element = dcc.Textarea(value=text,
                #                        style={'width': '100%'},
                #                        className='mb-2')
    return data_table, title, fig


@app.callback(
    Output('app-parameters', 'children'),
    Output('app-content', 'children'),
    Input('model-selection', 'value'),
    Input('action', 'value'),
    Input('jobs-table', 'selected_rows'),
    State('jobs-table', 'data'),
    prevent_intial_call=True)
def load_parameters_and_content(model_selection, action_selection, row, data_table):
    '''
    This callback dynamically populates the parameters and contents of the website according to the selected action &
    model.
    Args:
        model_selection:    Selected model (from content registry)
        action_selection:   Selected action (pre-defined actions in MLCoach)
        row:                Selected job (model)
        jobs-table:         Data in table of jobs
    Returns:
        app-parameters:     Parameters according to the selected model & action
        app-content:        Contents (right hand side) according to the selected model & action
    '''
    data = model_list_GET_call()
    if model_selection == 'tf-NN' and action_selection == 'train_model':
        conditions = {'model_name': 'Tensorflow-NN'}
        model = [d for d in data if all((k in d and d[k] == v) for k, v in conditions.items())]
        parameters = JSONParameterEditor(
            _id={'type': 'parameter_editor'},       # pattern match _id (base id), name
            json_blob=model[0]["gui_parameters"]
        )
        parameters.init_callbacks(app)
    if action_selection == 'evaluate_model':
        parameters = DATA_AUG_WIDGET.copy()
    if action_selection == 'prediction_model':
        parameters = DATA_AUG_WIDGET.copy()
    if action_selection == 'transfer_learning':
        num_layers = 0
        disable_comp = True
        try:
            if data_table[row[0]]['job_type'] == 'train_model':
                try:
                    logs = data_table[row[0]]['job_logs']
                    num_layers = int(logs[logs.find('Length: ')+8:logs.find(' layers')])
                    disable_comp = False
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)
        parameters = DATA_AUG_WIDGET.copy()
        init_layer = dbc.FormGroup([
            dbc.Label('Choose a trained model from the job list and select a layer to start training at below'),
            dcc.Slider(id='init-layer',
                       min=0,
                       max=num_layers,
                       disabled=disable_comp,
                       step=1,
                       value=0,
                       tooltip={'always_visible': True,
                                'placement': 'bottom'})
        ])
        parameters = parameters + EXTRA_PARAMETERS + [init_layer]
    contents = DATA_PREPROCESS_WIDGET.copy()
    return parameters, html.Div(contents)


@app.callback(
    [Output('img-output', 'figure'),
     Output('img-slider', 'max')],
    Input('img-slider', 'value'),
    State('action', 'value'),
    prevent_intial_call=True
)
def refresh_image(img_ind, action_selection):
    '''
    This callback updates the image in the display
    Args:
        img_ind:            Index of image according to the slider value
        action_selection:   Action selection (train vs test set)
    Returns:
        img-output:         Output figure
        img-slider-max:     Maximum value of the slider according to the dataset (train vs test)
    '''
    try:
        if action_selection in ['train_model', 'transfer_learning']:
            image = Image.open(TRAIN_DIR + '/' + list_train_filename[img_ind])
            slider_max = len(list_train_filename)-1
        else:
            image = Image.open(TEST_DIR + '/' + list_test_filename[img_ind])
            slider_max = len(list_test_filename)-1
    except Exception as e:
        print(e)
    fig = px.imshow(image) #, color_continuous_scale="gray")
    fig.update_xaxes(
        showgrid=False,
        showticklabels=False,
        zeroline=False
    )
    fig.update_yaxes(
        showgrid=False,
        showticklabels=False,
        zeroline=False
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=10))
    fig.update_traces(dict(showscale=False, coloraxis=None))
    return fig, slider_max


@app.callback(
    Output('dummy-output', 'children'),
    Input('execute', 'n_clicks'),
    [State('app-parameters', 'children'),
     State('action', 'value'),
     State('jobs-table', 'data'),
     State('jobs-table', 'selected_rows')],
    prevent_intial_call=True)
def execute(clicks, children, action_selection, job_data, row):
    '''
    This callback submits a job request to the compute service according to the selected action & model
    Args:
        clicks:             Execute button triggers this callback
        children:           Model parameters
        action_selection:   Action selected
        job_data:           Lists of jobs
        row:                Selected row (job)
    Returns:
        None
    '''
    if clicks > 0:
        contents = []
        experiment_id = str(uuid.uuid4())
        out_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER, experiment_id))
        out_path.mkdir(parents=True, exist_ok=True)
        input_params = {}
        if bool(children):
            try:
                for child in children['props']['children']:
                    key = child["props"]["children"][1]["props"]["id"]["param_key"]
                    value = child["props"]["children"][1]["props"]["value"]
                    input_params[key] = value
                # for count, child in enumerate(children['props']['children']):
                #     print(child)
                #     if count%2 == 1:
                #         key = child["props"]["id"]
                #         value = child["props"]["value"]
                #         input_params[key] = value
                # print(input_params)
            except Exception:
                for child in children:
                    key = child["props"]["children"][1]["props"]["id"]
                    value = child["props"]["children"][1]["props"]["value"]
                    input_params[key] = value
        json_dict = input_params
        if action_selection == 'train_model':
            command = "python3 src/train_model.py"
            directories = [TRAIN_DIR, VAL_DIR, str(out_path)]
        else:
            training_exp_id = job_data[row[0]]['experiment_id']
            in_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER, training_exp_id))
        if action_selection == 'evaluate_model':
            command = "python3 src/evaluate_model.py"
            directories = [TEST_DIR, str(in_path) + '/model.h5']
        if action_selection == 'prediction_model':
            command = "python3 src/predict_model.py"
            directories = [TEST_DIR, str(in_path) + '/model.h5', str(out_path)]
        if action_selection == 'transfer_learning':
            command = "python3 src/transfer_learning.py"
            directories = [TRAIN_DIR, VAL_DIR, str(in_path) + '/model.h5', str(out_path)]
        job = SimpleJob(user=USER,
                        job_type=action_selection,
                        description='',
                        deploy_location='local',
                        gpu=True,
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
    '''
    This callback plots the loss function
    Args:
        n:              Time interval that triggers this callback
    Returns:
        training_loss:  Loss plot
    '''
    job = get_job(USER, 'mlcoach')
    job = job[0]
    logs = job['container_logs']
    if logs:
        df = pd.read_csv(StringIO(logs), sep=' ')
        fig = px.line(df, x="epoch", y="loss")
        fig.update_traces(mode='markers+lines')
        return fig


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)#, dev_tools_ui=False)
