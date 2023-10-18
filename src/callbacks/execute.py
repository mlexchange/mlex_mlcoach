import json
import pathlib

from dash import Input, Output, State, callback
import dash

from file_manager.data_project import DataProject
from app_layout import USER, DATA_DIR
from utils.job_utils import MlexJob, TableJob
from utils.data_utils import prepare_directories


@callback(
    Output("resources-setup", "is_open"),
    Output("counters", "data"),
    Output("warning-cause-execute", "data"),

    Input("execute", "n_clicks"),
    Input("submit", "n_clicks"),

    State("app-parameters", "children"),
    State("num-cpus", "value"),
    State("num-gpus", "value"),
    State("action", "value"),
    State("jobs-table", "data"),
    State("jobs-table", "selected_rows"),
    State({'base_id': 'file-manager', 'name': 'docker-file-paths'},'data'),
    State("counters", "data"),
    State("model-name", "value"),
    State({'base_id': 'file-manager', 'name': 'project-id'}, 'data'),
    State("event-id", "value"),
    prevent_intial_call=True)
def execute(execute, submit, children, num_cpus, num_gpus, action_selection, job_data, row, file_paths,
            counters, model_name, project_id, event_id):
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
        file_paths:         Data project information
        counters:           List of counters to assign a number to each job according to its action 
                            (train vs evaluate)
        model_name:         User-defined name for training or prediction model
        project_id:         Data project id
        event_id:           Tagging event id for version control of tags
    Returns:
        open/close the resources setup modal, and submits the training/prediction job accordingly
    '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    data_project = DataProject()
    data_project.init_from_dict(file_paths)
    if 'execute.n_clicks' in changed_id:
        if len(data_project.data) == 0:
            return False, counters, 'no_dataset'
        if action_selection != 'train_model' and not row:
            return False, counters, 'no_row_selected'
        if row:
            if action_selection != 'train_model' and job_data[row[0]]['job_type'].split(' ')[0] != 'train_model':
                return False, counters, 'no_row_selected'
        return True, counters, ''
    if 'submit.n_clicks' in changed_id:
        counters = TableJob.get_counter(USER)
        experiment_id, out_path, data_info = prepare_directories(USER, data_project, project_id)
        input_params = {}
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
        if action_selection == 'train_model':
            counters[0] = counters[0] + 1
            count = counters[0]
            command = f"python3 src/train_model.py -d {data_info} -o {out_path} -e {event_id}"
        else:
            training_exp_id = job_data[row[0]]['experiment_id']
            model_path = pathlib.Path('/app/work/data/mlexchange_store/{}/{}'.format(USER, 
                                                                                     training_exp_id))
        if action_selection == 'prediction_model':
            counters[1] = counters[1] + 1
            count = counters[1]
            command = f"python3 src/predict_model.py -d {data_info} -m {model_path} -o {out_path}"
            kwargs = {'train_params': job_data[row[0]]['parameters']}
        if len(model_name)==0:      # if model_name was not defined
            model_name = f'{action_selection} {count}'
        job_kwargs = {'uri': 'mlexchange1/tensorflow-neural-networks:latest',
                      'type': 'docker',
                      'cmd': f"{command} -p \'{json.dumps(input_params)}\'",
                      'kwargs': 
                        {'job_type': action_selection,
                         'experiment_id': experiment_id,
                         'dataset': project_id,
                         'params': input_params,
                         **kwargs}
                        }
        job = MlexJob(service_type='backend',
                      description=model_name,
                      working_directory='{}'.format(DATA_DIR),
                      job_kwargs=job_kwargs)
        job.submit(USER, num_cpus, num_gpus)
        return False, counters, ''
    return False, counters, ''