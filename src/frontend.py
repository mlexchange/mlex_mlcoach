import shutil, json
import pathlib
from dash import Input, Output, State, dcc
from uuid import uuid4

from file_manager.data_project import DataProject
from app_layout import app, USER, DATA_DIR, long_callback_manager
from callbacks.display import refresh_image, refresh_results, toggle_warning_modal
from callbacks.download import toggle_storage_modal, disable_download
from callbacks.load_labels import load_from_splash_modal
from callbacks.execute import execute
from callbacks.table import update_table, delete_row
from dash_component_editor import JSONParameterEditor
from utils.job_utils import MlexJob
from utils.data_utils import prepare_directories, get_input_params
from utils.model_utils import get_gui_components, get_model_content


@app.callback(
    Output('app-parameters', 'children'),
    Input('model-selection', 'value'),
    Input('action', 'value'),
    prevent_intial_call=True
)
def load_parameters(model_selection, action_selection):
    '''
    This callback dynamically populates the parameters and contents of the website according to the 
    selected action & model.
    Args:
        model_selection:    Selected model (from content registry)
        action_selection:   Selected action (pre-defined actions in Data Clinic)
    Returns:
        app-parameters:     Parameters according to the selected model & action
    '''
    parameters = get_gui_components(model_selection, action_selection)
    gui_item = JSONParameterEditor(_id={'type': str(uuid4())}, # pattern match _id (base id), name
                                   json_blob=parameters,
                                   )
    gui_item.init_callbacks(app)
    return gui_item


@app.long_callback(
    Output("download-out", "data"),

    Input("download-button", "n_clicks"),
    
    State("jobs-table", "data"),
    State("jobs-table", "selected_rows"),
    manager=long_callback_manager,
    prevent_intial_call=True
)
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


@app.long_callback(
    Output("job-alert-confirm", "is_open"),

    Input("submit", "n_clicks"),

    State("app-parameters", "children"),
    State("num-cpus", "value"),
    State("num-gpus", "value"),
    State("action", "value"),
    State("jobs-table", "data"),
    State("jobs-table", "selected_rows"),
    State({'base_id': 'file-manager', 'name': 'docker-file-paths'},'data'),
    State("model-name", "value"),
    State({'base_id': 'file-manager', 'name': 'project-id'}, 'data'),
    State("event-id", "value"),
    State("model-selection", "value"),
    running=[
        (Output("job-alert", "is_open"), "True", "False")
    ],
    manager=long_callback_manager,
    prevent_initial_call=True
)
def submit_ml_job(submit, children, num_cpus, num_gpus, action_selection, job_data, row, file_paths,
            model_name, project_id, event_id, model_id):
    '''
    This callback submits a job request to the compute service according to the selected action & model
    Args:
        submit:             Submit button
        children:           Model parameters
        num_cpus:           Number of CPUs assigned to job
        num_gpus:           Number of GPUs assigned to job
        action_selection:   Action selected
        job_data:           Lists of jobs
        row:                Selected row (job)
        file_paths:         Data project information
        model_name:         User-defined name for training or prediction model
        project_id:         Data project id
        event_id:           Tagging event id for version control of tags
        model_id:           UID of model in content registry
    Returns:
        open the alert indicating that the job was submitted
    '''
    data_project = DataProject()
    data_project.init_from_dict(file_paths)
    model_uri, [train_cmd, prediction_cmd] = get_model_content(model_id)
    experiment_id, out_path, data_info = prepare_directories(USER, data_project, project_id)
    input_params = get_input_params(children)
    kwargs = {}
    if action_selection == 'train_model':
        command = f"{train_cmd} -d {data_info} -o {out_path} -e {event_id}"
    else:
        training_exp_id = job_data[row[0]]['experiment_id']
        model_path = pathlib.Path('/app/work/data/mlexchange_store/{}/{}'.format(USER, training_exp_id))
        command = f"{prediction_cmd} -d {data_info} -m {model_path} -o {out_path}"
        kwargs = {'train_params': job_data[row[0]]['parameters']}
    job = MlexJob(
        service_type='backend',
        description=model_name,
        working_directory='{}'.format(DATA_DIR),
        job_kwargs={
            'uri': model_uri,
            'type': 'docker',
            'cmd': f"{command} -p \'{json.dumps(input_params)}\'",
            'kwargs': {
                'job_type': action_selection,
                'experiment_id': experiment_id,
                'dataset': project_id,
                'params': input_params,
                **kwargs
                }
            }
        )
    job.submit(USER, num_cpus, num_gpus)
    return True


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0')