import json
import pathlib

import dash
from dash import Input, Output, State, callback

from file_manager.data_project import DataProject
from app_layout import USER, DATA_DIR
from utils.job_utils import MlexJob, TableJob
from utils.data_utils import prepare_directories, get_input_params
from utils.model_utils import get_model_content


@callback(
    Output("resources-setup", "is_open"),
    Output("warning-cause-execute", "data"),

    Input("execute", "n_clicks"),
    Input("submit", "n_clicks"),

    State("action", "value"),
    State("jobs-table", "data"),
    State("jobs-table", "selected_rows"),
    State({'base_id': 'file-manager', 'name': 'docker-file-paths'},'data'),
    prevent_initial_call=True)
def execute(execute, submit, action_selection, job_data, row, file_paths):
    '''
    This callback submits a job request to the compute service according to the selected action & model
    Args:
        execute:            Execute button
        submit:             Submit button
        action_selection:   Action selected
        job_data:           Lists of jobs
        row:                Selected row (job)
        file_paths:         Data project information
    Returns:
        open/close the resources setup modal, and submits the training/prediction job accordingly
        warning_cause:      Activates a warning pop-up window if needed
    '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    data_project = DataProject()
    data_project.init_from_dict(file_paths)
    if 'execute.n_clicks' in changed_id:
        if len(data_project.data) == 0:
            return False, 'no_dataset'
        elif action_selection != 'train_model' and not row:
            return False, 'no_row_selected'
        elif action_selection != 'train_model' and job_data[row[0]]['job_type']!='train_model':
            return False, 'no_row_selected'
        else:
            return True, ''
    else:
        return False, ''