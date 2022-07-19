import json
import os
import pathlib
import shutil
import zipfile

import dash
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_daq as daq
import dash_uploader as du
import numpy as np
import pandas as pd
import PIL.Image as Image
import plotly.graph_objects as go
import uuid
import requests

from file_manager import filename_list, move_a_file, move_dir, add_paths_from_dir, \
                         check_duplicate_filename, docker_to_local_path, local_to_docker_path, \
                         file_explorer, DOCKER_DATA, DOCKER_HOME, LOCAL_HOME, UPLOAD_FOLDER_ROOT
from helpers import SimpleJob
from helpers import get_job, generate_figure, get_class_prob, model_list_GET_call, plot_figure, get_gui_components,\
    init_counter
from kwarg_editor import JSONParameterEditor
import templates

external_stylesheets = [dbc.themes.BOOTSTRAP, "../assets/segmentation-style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# Global variables
DATA_DIR = str(os.environ['DATA_DIR'])
USER = 'admin'
MODELS = model_list_GET_call()


RESOURCES_SETUP = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader("Resources Setup"),
                dbc.ModalBody(
                    children=[
                        dbc.FormGroup([
                                dbc.Label('Number of CPUs'),
                                dbc.Input(id='num-cpus',
                                          type="int",
                                          value=1)]),
                        dbc.FormGroup([
                                dbc.Label('Number of GPUs'),
                                dbc.Input(id='num-gpus',
                                          type="int",
                                          value=0)])
                    ]),
                dbc.ModalFooter(
                    dbc.Button(
                        "Submit Job", id="submit", className="ms-auto", n_clicks=0
                    )
                ),
            ],
            id="resources-setup",
            centered=True,
            is_open=False,
        ),
    ]
)


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
                            {'name': 'Parameters', 'id': 'parameters'},
                            {'name': 'Experiment ID', 'id': 'experiment_id'},
                            {'name': 'Dataset', 'id': 'dataset'},
                            {'name': 'Logs', 'id': 'job_logs'}
                        ],
                        data=[],
                        hidden_columns=['job_id', 'experiment_id', 'dataset'],
                        row_selectable='single',
                        style_cell={'padding': '1rem',
                                    'textAlign': 'left',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',
                                    'maxWidth': 0},
                        fixed_rows={'headers': True},
                        css=[{"selector": ".show-hide", "rule": "display: none"}],
                        style_data_conditional=[
                            {'if': {'column_id': 'status', 'filter_query': '{status} = complete'},
                             'backgroundColor': 'green',
                             'color': 'white'},
                            {'if': {'column_id': 'status', 'filter_query': '{status} = failed'},
                             'backgroundColor': 'red',
                             'color': 'white'}
                        ],
                        style_table={'height': '30rem', 'overflowY': 'auto', 'overflowX': 'scroll'}
                    )
                ],
            ),
        dbc.Modal([
            dbc.ModalHeader("Job Logs"),
            dbc.ModalBody(id='log-display'),
            dbc.ModalFooter(dbc.Button("Close", id="modal-close", className="ml-auto")),
            ],
            id='log-modal',
            size='xl')
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
                            # {'label': 'Evaluate Model on Data', 'value': 'evaluate_model'},
                            {'label': 'Test Prediction using Model', 'value': 'prediction_model'},
                            # {'label': 'Transfer Learning', 'value': 'transfer_learning'},
                        ],
                        value='train_model')
                ]),
                dbc.FormGroup([
                    dbc.Label('Model'),
                    dcc.Dropdown(
                        id='model-selection',
                        options=MODELS,
                        value=MODELS[0]['value'])
                ]),
                dbc.FormGroup([
                    dbc.Label('Data'),
                    file_explorer,
                ])
            ])
        ]
    ),
    dbc.Card(
        children=[
            dbc.CardHeader("Parameters"),
            dbc.CardBody(html.Div(id='app-parameters'))
        ]
    ),
    dbc.Modal(
        [
            dbc.ModalHeader("Warning"),
            dbc.ModalBody(id="warning-msg"),
            dbc.ModalFooter([
                dbc.Button(
                    "OK", id="ok-button", color='danger', outline=False,
                    className="ms-auto", n_clicks=0
                ),
            ]),
        ],
        id="warning-modal",
        is_open=False,
    ),
    dcc.Store(id='warning-cause', data=''),
    dcc.Store(id='counter', data=init_counter(USER))
]

# App contents (right hand side)
CONTENT = [
    html.Div([dbc.Row([
        dbc.Col(dbc.Card(
            children=[dbc.CardHeader('Data Overview'),
                      dbc.CardBody(children=[
                          html.Div(
                              id='app-content',
                              children = [dcc.Graph(id='img-output'),
                                          html.Output(id='label-output',
                                                      style={'height': '2rem', 'overflow': 'hidden',
                                                             'text-overflow': 'hidden'}),
                                          dbc.Label(id='current-image-label'),
                                          dcc.Slider(id='img-slider',
                                                     min=0,
                                                     value=0,
                                                     tooltip={'always_visible': True, 'placement': 'bottom'}),
                                          dbc.Button('Execute',
                                                     id='execute',
                                                     n_clicks=0,
                                                     className='m-1',
                                                     style={'width': '100%', 'justify-content': 'center'})
                                          ],
                              style={'display': 'none'}),
                      ], style={'height': '32rem'})
                      ]),
            width=5),
        dbc.Col(dbc.Card(
            id = 'results',
            children=[dbc.CardHeader('Results'),
                      dbc.CardBody(children = [dcc.Graph(id='results-plot',
                                                        style={'display': 'none'}),
                                               dcc.Textarea(id='results-text',
                                                            style={'display': 'none'},
                                                            className='mb-2'),
                                               dbc.Button('Download Results',
                                                          id='download-button',
                                                          n_clicks=0,
                                                          className='m-1',
                                                          style={'display': 'None'}),
                                               dcc.Download(id='download-out')
                                               ],
                                   style={'height': '32rem'})]),
            width=7)]),
        dcc.Interval(id='interval', interval=5 * 1000, n_intervals=0)
    ]),
    JOB_STATUS
]

# Setting up initial webpage layout
app.title = 'MLCoach'
app._favicon = 'mlex.ico'
du.configure_upload(app, UPLOAD_FOLDER_ROOT, use_upload_id=False)
app.layout = html.Div([templates.header(),
                       dbc.Container([
                           dbc.Row([dbc.Col(SIDEBAR, width=3),
                                    dbc.Col(CONTENT,
                                            width=9,
                                            style={'align-items': 'center', 'justify-content': 'center'}),
                                    html.Div(id='dummy-output')
                                   ]),
                           RESOURCES_SETUP],
                           fluid=True
                       )])


@app.callback(
    Output("collapse", "is_open"),

    Input("collapse-button", "n_clicks"),
    Input("import-dir", "n_clicks"),

    State("collapse", "is_open")
)
def toggle_collapse(collapse_button, import_button, is_open):
    '''
    This callback toggles the file manager
    Args:
        collapse_button:    "Open File Manager" button
        import_button:      Import button
        is_open:            Open/close File Manager modal state
    '''
    if collapse_button or import_button:
        return not is_open
    return is_open


@app.callback(
    Output("warning-modal", "is_open"),
    Output("warning-msg", "children"),

    Input("warning-cause", "data"),
    Input("ok-button", "n_clicks"),

    State("warning-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_warning_modal(warning_cause, ok_n_clicks, is_open):
    '''
    This callback toggles a warning/error message
    Args:
        warning_cause:      Cause that triggered the warning
        ok_n_clicks:        Close the warning
        is_open:            Close/open state of the warning
    '''
    if ok_n_clicks:
        return not is_open, ""
    if warning_cause == 'wrong_dataset':
        return not is_open, "The dataset you have selected is not supported. Please select (1) a data directory " \
                        "where each subfolder corresponds to a given category, OR (2) an NPZ file."
    if warning_cause == 'different_size':
        return not is_open, "The number of images and labels do not match. Please select a different dataset."
    else:
        return not is_open, ""


@app.callback(
    Output("modal", "is_open"),

    Input("delete-files", "n_clicks"),
    Input("confirm-delete", "n_clicks"),

    State("modal", "is_open")
)
def toggle_modal(n1, n2, is_open):
    '''
    This callback toggles a confirmation message for file manager
    Args:
        n1:         Delete files button
        n2:         Confirm delete button
        is_open:    Open/close confirmation modal state
    '''
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("npz-modal", "is_open"),
    Output("npz-img-key", "options"),
    Output("npz-label-key", "options"),

    Input("import-dir", "n_clicks"),
    Input("confirm-import", "n_clicks"),
    Input("npz-img-key", "value"),
    Input("npz-label-key", "value"),

    State("npz-modal", "is_open"),
    State("docker-file-paths", "data"),
)
def toggle_modal_keyword(import_button, confirm_import, img_key, label_key, is_open, npz_path):
    '''
    This callback opens the modal to select the keywords within the NPZ file. When a keyword is selected for images or
    labels, this option is removed from the options of the other.
    Args:
        import_button:      Import button
        confirm_import:     Confirm import button
        img_key:            Selected keyword for the images
        label_key:          Selected keyword for the labels
        is_open:            Open/close status of the modal
        npz_path:           Path to NPZ file
    Returns:
        toggle_modal:       Open/close modal
        img_options:        Keyword options for images
        label_options:      Keyword options for labels
    '''
    img_options = []
    label_options = []
    toggle_modal = is_open
    changed_id = dash.callback_context.triggered[0]['prop_id']
    if npz_path:
        if npz_path[0].split('.')[-1] == 'npz':
            data = np.load(npz_path[0])
            img_key_list = list(data.keys())
            label_key_list = list(data.keys())
            # if this value has been previously selected, it is removed from its options
            if label_key in img_key_list:
                img_key_list.remove(label_key)
            df_img = pd.DataFrame({'c': img_key_list})
            if img_key in label_key_list:
                label_key_list.remove(img_key)
            df_label = pd.DataFrame({'c': label_key_list})
            img_options = [{'label':i, 'value':i} for i in df_img['c']]
            label_options = [{'label':i, 'value':i} for i in df_label['c']]
            toggle_modal = True
    if is_open and 'confirm-import.n_clicks' in changed_id:
        toggle_modal = False
    return toggle_modal, img_options, label_options


@app.callback(
    Output('dummy-data', 'data'),

    Input('dash-uploader', 'isCompleted'),

    State('dash-uploader', 'fileNames')
)
def upload_zip(iscompleted, upload_filename):
    '''
    This callback uploads a ZIP file
    Args:
        iscompleted:        The upload operation is completed (bool)
        upload_filename:    Filename of the uploaded content
    '''
    if not iscompleted:
        return 0
    if upload_filename is not None:
        path_to_zip_file = pathlib.Path(UPLOAD_FOLDER_ROOT) / upload_filename[0]
        if upload_filename[0].split('.')[-1] == 'zip':  # unzip files and delete zip file
            zip_ref = zipfile.ZipFile(path_to_zip_file)  # create zipfile object
            path_to_folder = pathlib.Path(UPLOAD_FOLDER_ROOT) / upload_filename[0].split('.')[-2]
            if (upload_filename[0].split('.')[-2] + '/') in zip_ref.namelist():
                zip_ref.extractall(pathlib.Path(UPLOAD_FOLDER_ROOT))  # extract file to dir
            else:
                zip_ref.extractall(path_to_folder)
            zip_ref.close()  # close file
            os.remove(path_to_zip_file)
    return 0


@app.callback(
    Output('files-table', 'data'),
    Output('docker-file-paths', 'data'),
    Output('data-path', 'data'),

    Input('browse-format', 'value'),
    Input('browse-dir', 'n_clicks'),
    Input('import-dir', 'n_clicks'),
    Input('confirm-delete', 'n_clicks'),
    Input('move-dir', 'n_clicks'),
    Input('files-table', 'selected_rows'),
    Input('data-path', 'data'),
    Input('import-format', 'value'),
    Input('my-toggle-switch', 'value'),
    Input('jobs-table', 'selected_rows'),
    Input("clear-data", "n_clicks"),
    Input("refresh-data", "n_clicks"),

    State('dest-dir-name', 'value'),
    State('jobs-table', 'data')
)
def file_manager(browse_format, browse_n_clicks, import_n_clicks, delete_n_clicks, move_dir_n_clicks, rows,
                 selected_paths, import_format, docker_path, job_rows, clear_data, refresh_data, dest, job_data):
    '''
    This callback displays manages the actions of file manager
    Args:
        browse_format:      File extension to browse
        browse_n_clicks:    Browse button
        import_n_clicks:    Import button
        delete_n_clicks:    Delete button
        move_dir_n_clicks:  Move button
        rows:               Selected rows
        selected_paths:     Selected paths in cache
        import_format:      File extension to import
        docker_path:        [bool] docker vs local path
        job_rows:           Selected rows in job table. If it's not a "training" model, it will load its results
                            instead of the data uploaded through File Manager. This is so that the user can observe
                            previous evaluation results
        clear_data:         Clear the loaded images
        refresh_data:       Refresh the loaded images 
        dest:               Destination path
        job_data:           Data in job table
    Returns
        files:              Filenames to be displayed in File Manager according to browse_format from docker/local path
        list_filename:      List of selected filenames in the directory AND SUBDIRECTORIES FROM DOCKER PATH
        selected_files:     List of selected filename FROM DOCKER PATH (no subdirectories)
    '''
    changed_id = dash.callback_context.triggered[0]['prop_id']
    # if a previous job is selected, it's data is automatically plotted
    if 'jobs-table.selected_rows' in changed_id and job_rows is not None:
        if job_data[job_rows[0]]["job_type"].split()[-1] != 'train_model':
            filenames = add_paths_from_dir(job_data[job_rows[0]]["dataset"], ['tiff', 'tif', 'jpg', 'jpeg', 'png'], [])
            return dash.no_update, filenames, dash.no_update

    supported_formats = []
    import_format = import_format.split(',')
    if import_format[0] == '*':
        supported_formats = ['tiff', 'tif', 'jpg', 'jpeg', 'png']
    else:
        for ext in import_format:
            supported_formats.append(ext.split('.')[1])

    files = []
    if browse_n_clicks or import_n_clicks:
        files = filename_list(DOCKER_DATA, browse_format)

    selected_files = []
    list_filename = []
    if bool(rows):
        for row in rows:
            file_path = files[row]
            selected_files.append(file_path)
            if file_path['file_type'] == 'dir':
                list_filename = add_paths_from_dir(file_path['file_path'], supported_formats, list_filename)
            else:
                list_filename.append(file_path['file_path'])

    if browse_n_clicks and changed_id == 'confirm-delete.n_clicks':
        for filepath in selected_files:
            if os.path.isdir(filepath['file_path']):
                shutil.rmtree(filepath['file_path'])
            else:
                os.remove(filepath['file_path'])
        selected_files = []
        files = filename_list(DOCKER_DATA, browse_format)

    if browse_n_clicks and changed_id == 'move-dir.n_clicks':
        if dest is None:
            dest = ''
        destination = DOCKER_DATA / dest
        destination.mkdir(parents=True, exist_ok=True)
        if bool(rows):
            sources = selected_paths
            for source in sources:
                if os.path.isdir(source['file_path']):
                    move_dir(source['file_path'], str(destination))
                    shutil.rmtree(source['file_path'])
                else:
                    move_a_file(source['file_path'], str(destination))
            selected_files = []
            files = filename_list(DOCKER_DATA, browse_format)
    if not docker_path:
        files = docker_to_local_path(files, DOCKER_HOME, LOCAL_HOME)
    
    if changed_id == 'refresh-data.n_clicks':
        list_filename, selected_files = [], []
        datapath = requests.get(f'http://labelmaker-api:8005/api/v0/export/datapath').json()
        if bool(datapath['datapath']) and os.path.isdir(datapath['datapath'][0]['file_path']):
            list_filename, selected_files = datapath['filenames'], datapath['datapath'][0]['file_path']
        return files,  list_filename, selected_files
        
    elif changed_id == 'import-dir.n_clicks':
        return files, list_filename, selected_files
        
    elif changed_id == 'clear-data.n_clicks':
        return [], [], []

    else:
        return files, dash.no_update, dash.no_update


@app.callback(
    Output('jobs-table', 'data'),
    Output('results-plot', 'figure'),
    Output('results-plot', 'style'),
    Output('results-text', 'value'),
    Output('results-text', 'style'),
    Output('log-modal', 'is_open'),
    Output('log-display', 'children'),
    Output('jobs-table', 'active_cell'),

    Input('interval', 'n_intervals'),
    Input('jobs-table', 'selected_rows'),
    Input('jobs-table', 'active_cell'),
    Input('img-slider', 'value'),
    Input('modal-close', 'n_clicks'),

    State('docker-file-paths', 'data'),
    State('jobs-table', 'data'),
    State('results-plot', 'figure'),
    prevent_initial_call=True
)
def update_table(n, row, active_cell, slider_value, close_clicks, filenames, current_job_table, current_fig):
    '''
    This callback updates the job table, loss plot, and results according to the job status in the compute service.
    Args:
        n:              Time intervals that trigger this callback
        row:            Selected row (job)
        slider_value:   Image slider value (current image)
        filenames:      Selected data files
        current_job_table:  Current job table
        current_fig:        Current loss plot
    Returns:
        jobs-table:     Updates the job table
        show-plot:      Shows/hides the loss plot
        loss-plot:      Updates the loss plot according to the job status (logs)
        results:        Testing results (probability)
    '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'modal-close.n_clicks' in changed_id:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, False, dash.no_update, None
    job_list = get_job(USER, 'mlcoach')
    data_table = []
    if job_list is not None:
        for job in job_list:
            params = str(job['job_kwargs']['kwargs']['params'])
            if job['job_kwargs']['kwargs']['job_type'].split(' ')[0] != 'train_model':
                params = params + '\nTraining Parameters: ' + str(job['job_kwargs']['kwargs']['train_params'])
            data_table.insert(0,
                              dict(
                                  job_id=job['uid'],
                                  job_type=job['job_kwargs']['kwargs']['job_type'],
                                  status=job['status']['state'],
                                  parameters=params,
                                  experiment_id=job['job_kwargs']['kwargs']['experiment_id'],
                                  job_logs=job['logs'],
                                  dataset=job['job_kwargs']['kwargs']['dataset'])
                              )
    is_open = dash.no_update
    log_display = dash.no_update
    if active_cell:
        row_log = active_cell["row"]
        col_log = active_cell["column_id"]
        if col_log == 'job_logs':       # show job logs
            is_open = True
            log_display = dcc.Textarea(value=data_table[row_log]["job_logs"],
                                       style={'width': '100%', 'height': '30rem', 'font-family':'monospace'})
        if col_log == 'parameters':     # show job parameters
            is_open = True
            log_display = dcc.Textarea(value=str(job['job_kwargs']['kwargs']['params']),
                                       style={'width': '100%', 'height': '30rem', 'font-family': 'monospace'})
    style_fig = {'display': 'none'}
    style_text = {'display': 'none'}
    val = ''
    fig = go.Figure(go.Scatter(x=[], y=[]))
    if row:
        log = data_table[row[0]]["job_logs"]
        if log:
            if data_table[row[0]]['job_type'].split(' ')[0] == 'train_model':
                start = log.find('epoch')
                if start > -1 and len(log) > start + 5:
                    fig = generate_figure(log, start)
                    style_fig = {'width': '100%', 'display': 'block'}
            if data_table[row[0]]['job_type'].split(' ')[0] == 'evaluate_model':
                val = log
                style_text = {'width': '100%', 'display': 'block'}
            if data_table[row[0]]['job_type'].split(' ')[0] == 'prediction_model':
                start = log.find('filename')
                if start > -1 and len(log) > start + 10:
                    fig = get_class_prob(log, start, filenames[slider_value])
                    style_fig = {'width': '100%', 'display': 'block'}
    if current_fig:
        try:
            print('Change?')
            if current_fig['data'][0]['y'] == list(fig['data'][0]['y']):
                print('No')
                fig = dash.no_update
            else:
                print('Yes')
        except Exception as e:
            print(e)
    if data_table == current_job_table:
        data_table = dash.no_update
    return data_table, fig, style_fig, val, style_text, is_open, log_display, None


@app.callback(
    Output("app-parameters", "children"),
    Output("download-button", "style"),

    Input("model-selection", "value"),
    Input("action", "value"),
    Input("jobs-table", "selected_rows"),
    prevent_intial_call=True)
def load_parameters(model_selection, action_selection, row):
    '''
    This callback dynamically populates the parameters of the website according to the selected action & model.
    Args:
        model_selection:    Selected model (from content registry)
        action_selection:   Selected action (pre-defined actions in MLCoach)
        row:                Selected job (model)
    Returns:
        app-parameters:     Parameters according to the selected model & action
        download-button:    Shows the download button
    '''
    parameters = get_gui_components(model_selection, action_selection)
    gui_item = JSONParameterEditor(_id={'type': 'parameter_editor'},  # pattern match _id (base id), name
                                   json_blob=parameters)
    gui_item.init_callbacks(app)
    style = dash.no_update
    if row is not None:
        style = {'width': '100%', 'justify-content': 'center'}
    return gui_item, style


@app.callback(
    Output("img-output", "figure"),
    Output("current-image-label", 'children'),
    Output("label-output", "children"),
    Output("img-slider", "max"),
    Output("img-slider", "value"),
    Output("app-content", "style"),
    Output("warning-cause", "data"),

    Input("import-dir", "n_clicks"),
    Input("confirm-import", "n_clicks"),
    Input("img-slider", "value"),
    Input("docker-file-paths", "data"),
    
    State("npz-img-key", "value"),
    State("npz-label-key", "value"),
    State("npz-modal", "is_open"),
    prevent_intial_call=True
)
def refresh_image(import_dir, confirm_import, img_ind, filenames, img_keyword, label_keyword, npz_modal):
    '''
    This callback updates the image in the display
    Args:
        import_dir:         Import button
        confirm_import:     Confirm import button
        img_ind:            Index of image according to the slider value
        filenames:          Selected data files
        jobs-table:         Data in table of jobs
        img_keyword:        Keyword for images in NPZ file
        label_keyword:      Keyword for labels in NPZ file
        npz_modal:          Open/close status of NPZ modal
    Returns:
        img-output:         Output figure
        label-output:       Output label
        img-slider-max:     Maximum value of the slider according to the dataset (train vs test)
        img-slider-value:   Current value of the slider
        content_style:      Content visibility
        warning-cause:      Cause that triggered warning pop-up
    '''
    current_im_label=''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if len(filenames)>0 and not npz_modal:
        try:
            if filenames[0].split('.')[-1] == 'npz':        # npz file
                if img_keyword is not None and label_keyword is not None:
                    data_npz = np.load(filenames[0])
                    data_npy = np.squeeze(data_npz[img_keyword])
                    label_npy = np.squeeze(data_npz[label_keyword])
                    if len(data_npy) != len(label_npy):
                        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 'different_size'
                    slider_max = len(data_npy) - 1
                    if img_ind>slider_max:
                        img_ind = 0
                    fig = plot_figure(data_npy[img_ind])
                    current_im_label = f"Image: {filenames[0]}"
                    label = f"Label: {label_npy[img_ind]}"
                else:
                    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, {'display': 'None'}, dash.no_update
            else:                                           # directory
                slider_max = len(filenames)-1
                if img_ind>slider_max:
                    img_ind = 0
                image = Image.open(filenames[img_ind])
                current_image_label = filenames[img_ind]
                fig = plot_figure(image)
                current_im_label = f"Image: {filenames[img_ind]}"
                label = f"Label: {filenames[img_ind].split('/')[-2]}"  # determined by the last directory in the path
            return fig, current_im_label, label, slider_max, img_ind, {'display': 'block'}, dash.no_update
        except Exception as e:
            print(f'Exception in refresh_image callback {e}')
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, {'display': 'None'}, 'wrong_dataset'
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, {'display': 'None'}, dash.no_update


@app.callback(
    Output("resources-setup", "is_open"),
    Output("counter", "data"),

    Input("execute", "n_clicks"),
    Input("submit", "n_clicks"),

    State("app-parameters", "children"),
    State("num-cpus", "value"),
    State("num-gpus", "value"),
    State("action", "value"),
    State("jobs-table", "data"),
    State("jobs-table", "selected_rows"),
    State('data-path', 'data'),
    State("docker-file-paths", "data"),
    State("counter", "data"),
    State("npz-img-key", "value"),
    State("npz-label-key", "value"),
    prevent_intial_call=True)
def execute(execute, submit, children, num_cpus, num_gpus, action_selection, job_data, row, data_path, filenames,
            counters, x_key, y_key):
    '''
    This callback submits a job request to the compute service according to the selected action & model
    Args:
        execute:            Execute button
        submit:             Submit button
        children:           Model parameters
        num_cpus:           Number of CPUs assigned to job
        num_gpus:           Number of GPUs assigned to job
        action_selection:   Action selected
        job_data:           Lists of jobs
        row:                Selected row (job)
        data_path:          Local path to data
        filenames:          Filenames in dataset
        counters:           List of counters to assign a number to each job according to its action (train vs evaluate)
        x_key:              Keyword for x data in NPZ file
        y_key:              Keyword for y data in NPZ file
    Returns:
        open/close the resources setup modal
    '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'execute.n_clicks' in changed_id:
        return True, counters
    if 'submit.n_clicks' in changed_id:
        experiment_id = str(uuid.uuid4())
        out_path = pathlib.Path('/app/work/data/mlexchange_store/{}/{}'.format(USER, experiment_id))
        out_path.mkdir(parents=True, exist_ok=True)
        input_params = {'x_key': x_key, 'y_key': y_key}
        kwargs = {}
        if bool(children):
            try:
                for child in children['props']['children']:
                    key = child["props"]["children"][1]["props"]["id"]["param_key"]
                    value = child["props"]["children"][1]["props"]["value"]
                    input_params[key] = value
            except Exception:
                for child in children:
                    key = child["props"]["children"][1]["props"]["id"]
                    value = child["props"]["children"][1]["props"]["value"]
                    input_params[key] = value
        try:
            data_path = data_path[0]['file_path']
        except Exception as e:
            print(e)
        if action_selection == 'train_model':
            counters[0] = counters[0] + 1
            count = counters[0]
            command = "python3 src/train_model.py"
            directories = [data_path, str(out_path)]
        else:
            training_exp_id = job_data[row[0]]['experiment_id']
            in_path = pathlib.Path('/app/work/data/mlexchange_store/{}/{}'.format(USER, training_exp_id))
        if action_selection == 'evaluate_model':
            counters[1] = counters[1] + 1
            count = counters[1]
            command = "python3 src/evaluate_model.py"
            directories = [data_path, str(in_path) + '/model.h5']
        if action_selection == 'prediction_model':
            counters[2] = counters[2] + 1
            count = counters[2]
            command = "python3 src/predict_model.py"
            kwargs = {'train_params': job_data[row[0]]['parameters']}
            directories = [data_path, str(in_path) + '/model.h5', str(out_path)]
        if action_selection == 'transfer_learning':
            counters[3] = counters[3] + 1
            count = counters[3]
            command = "python3 src/transfer_learning.py"
            directories = [data_path, str(in_path) + '/model.h5', str(out_path)]
        job = SimpleJob(service_type='backend',
                        working_directory='{}'.format(DATA_DIR),
                        uri='mlexchange1/tensorflow-neural-networks',
                        cmd=' '.join([command] + directories + ['\'' + json.dumps(input_params) + '\'']),
                        kwargs={'job_type': f'{action_selection} {count}',
                                'experiment_id': experiment_id,
                                'dataset': data_path,
                                'params': input_params,
                                **kwargs})
        job.submit(USER, num_cpus, num_gpus)
        return False, counters
    return False, counters


@app.callback(
    Output("download-out", "data"),
    Input("download-button", "n_clicks"),
    State("jobs-table", "data"),
    State("jobs-table", "selected_rows"),
    prevent_intial_call=True)
def save_results(download, job_data, row):
    '''
    This callback saves the experimental results as a ZIP file
    Args:
        download:   Download button
        job_data:   Table of jobs
        row:        Selected job/row
    Returns:
        ZIP file with results
    '''
    if download and row:
        experiment_id = job_data[row[0]]["experiment_id"]
        experiment_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER, experiment_id))
        shutil.make_archive('/app/tmp/results', 'zip', experiment_path)
        return dcc.send_file('/app/tmp/results.zip')
    else:
        return None


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8062)#, dev_tools_ui=False)
