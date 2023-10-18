import shutil
import pathlib
import dash
from dash import Input, Output, State, dcc
from uuid import uuid4

from app_layout import app, USER, long_callback_manager
from callbacks.display import refresh_image, refresh_results, toggle_warning_modal
from callbacks.load_labels import load_from_splash_modal
from callbacks.execute import execute
from callbacks.table import update_table, delete_row
from dash_component_editor import JSONParameterEditor
from utils.model_utils import get_gui_components


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
    gui_item = JSONParameterEditor(_id={'type': str(uuid4())},  # pattern match _id (base id), name
                                   json_blob=parameters)
    gui_item.init_callbacks(app)
    style = dash.no_update
    if row is not None:
        style = {'width': '100%', 'justify-content': 'center'}
    return gui_item, style


@app.long_callback(
    Output("download-out", "data"),

    Input("download-button", "n_clicks"),
    
    State("jobs-table", "data"),
    State("jobs-table", "selected_rows"),
    manager=long_callback_manager,
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


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0')
