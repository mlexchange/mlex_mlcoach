import json
import os
import pathlib
import pickle
import shutil
import tempfile
from uuid import uuid4

from dash import Input, Output, State, dcc
from file_manager.data_project import DataProject

from app_layout import DATA_DIR, USER, app, long_callback_manager
from callbacks.display import (  # noqa: F401
    close_warning_modal,
    open_warning_modal,
    refresh_image,
    refresh_label,
    refresh_results,
    update_slider_boundaries_new_dataset,
    update_slider_boundaries_prediction,
    update_slider_value,
)
from callbacks.download import disable_download, toggle_storage_modal  # noqa: F401
from callbacks.execute import close_resources_popup, execute  # noqa: F401
from callbacks.load_labels import load_from_splash_modal  # noqa: F401
from callbacks.table import delete_row, open_job_modal, update_table  # noqa: F401
from dash_component_editor import JSONParameterEditor
from utils.data_utils import get_input_params, prepare_directories
from utils.job_utils import MlexJob
from utils.model_utils import get_gui_components, get_model_content

DIR_MOUNT = os.getenv("DIR_MOUNT", "/data")


app.clientside_callback(
    """
    function(n) {
        if (typeof Intl === 'object' && typeof Intl.DateTimeFormat === 'function') {
            const dtf = Intl.DateTimeFormat();
            if (typeof dtf === 'object' && typeof dtf.resolvedOptions === 'function') {
                const ro = dtf.resolvedOptions();
                if (typeof ro === 'object' && typeof ro.timeZone === 'string') {
                    return ro.timeZone;
                }
            }
        }
        return 'Timezone information not available';
    }
    """,
    Output("timezone-browser", "value"),
    Input("interval", "n_intervals"),
)


@app.callback(
    Output("app-parameters", "children"),
    Input("model-selection", "value"),
    Input("action", "value"),
    prevent_intial_call=True,
)
def load_parameters(model_selection, action_selection):
    """
    This callback dynamically populates the parameters and contents of the website according to the
    selected action & model.
    Args:
        model_selection:    Selected model (from content registry)
        action_selection:   Selected action (pre-defined actions in Data Clinic)
    Returns:
        app-parameters:     Parameters according to the selected model & action
    """
    parameters = get_gui_components(model_selection, action_selection)
    gui_item = JSONParameterEditor(
        _id={"type": str(uuid4())},  # pattern match _id (base id), name
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
    prevent_intial_call=True,
)
def save_results(download, job_data, row):
    """
    This callback saves the experimental results as a ZIP file
    Args:
        download:   Download button
        job_data:   Table of jobs
        row:        Selected job/row
    Returns:
        ZIP file with results
    """
    if download and row:
        experiment_id = job_data[row[0]]["experiment_id"]
        experiment_path = pathlib.Path(f"{DATA_DIR}/mlex_store/{USER}/{experiment_id}")
        with tempfile.TemporaryDirectory():
            tmp_dir = tempfile.gettempdir()
            archive_path = os.path.join(tmp_dir, "results")
            shutil.make_archive(archive_path, "zip", experiment_path)
        return dcc.send_file(f"{archive_path}.zip")
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
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    State("model-name", "value"),
    State("event-id", "value"),
    State("model-selection", "value"),
    State({"base_id": "file-manager", "name": "log-toggle"}, "on"),
    State("img-labeled-indx", "options"),
    running=[(Output("job-alert", "is_open"), "True", "False")],
    manager=long_callback_manager,
    prevent_initial_call=True,
)
def submit_ml_job(
    submit,
    children,
    num_cpus,
    num_gpus,
    action_selection,
    job_data,
    row,
    data_project_dict,
    model_name,
    event_id,
    model_id,
    log,
    labeled_dropdown,
):
    """
    This callback submits a job request to the compute service according to the selected action & model
    Args:
        submit:             Submit button
        children:           Model parameters
        num_cpus:           Number of CPUs assigned to job
        num_gpus:           Number of GPUs assigned to job
        action_selection:   Action selected
        job_data:           Lists of jobs
        row:                Selected row (job)
        data_project_dict:  Data project dictionary
        model_name:         User-defined name for training or prediction model
        event_id:           Tagging event id for version control of tags
        model_id:           UID of model in content registry
        log:                Log toggle
        labeled_dropdown:   Indexes of the labeled images in this data set
    Returns:
        open the alert indicating that the job was submitted
    """
    # Get model information from content registry
    model_uri, [train_cmd, prediction_cmd] = get_model_content(model_id)

    # Get model parameters
    input_params = get_input_params(children)
    input_params["log"] = log

    kwargs = {}
    data_project = DataProject.from_dict(data_project_dict)

    if action_selection == "train_model":
        experiment_id, out_path, data_info = prepare_directories(
            USER, data_project, labeled_indices=labeled_dropdown
        )
        command = f"{train_cmd} -d {data_info} -o {out_path} -e {event_id}"
    else:
        experiment_id, out_path, data_info = prepare_directories(
            USER, data_project, train=False
        )
        training_exp_id = job_data[row[0]]["experiment_id"]
        model_path = pathlib.Path(f"{DATA_DIR}/mlex_store/{USER}/{training_exp_id}")
        command = f"{prediction_cmd} -d {data_info} -m {model_path} -o {out_path}"
        kwargs = {"train_params": job_data[row[0]]["parameters"]}

        with open(f"{out_path}/.file_manager_vars.pkl", "wb") as file:
            pickle.dump(
                data_project_dict,
                file,
            )

    # Define MLExjob
    job = MlexJob(
        service_type="backend",
        description=model_name,
        working_directory="{}".format(DIR_MOUNT),
        job_kwargs={
            "uri": model_uri,
            "type": "docker",
            "cmd": f"{command} -p '{json.dumps(input_params)}'",
            "kwargs": {
                "job_type": action_selection,
                "experiment_id": experiment_id,
                "dataset": data_project.project_id,
                "params": input_params,
                **kwargs,
            },
        },
    )

    # Submit job
    job.submit(USER, num_cpus, num_gpus)
    return True


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")
