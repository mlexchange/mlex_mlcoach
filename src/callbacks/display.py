import os
import pathlib
import pickle
import time
import traceback

import dash
import pandas as pd
import requests
from dash import Input, Output, State, callback
from dash.exceptions import PreventUpdate
from file_manager.data_project import DataProject

from src.app_layout import DATA_DIR, SPLASH_URL, TILED_KEY, USER, logger
from src.utils.job_utils import TableJob
from src.utils.plot_utils import generate_loss_plot, get_class_prob, plot_figure


@callback(
    Output("img-output-store", "data"),
    Output("img-uri", "data"),
    Input("img-slider", "value"),
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    State("jobs-table", "selected_rows"),
    State("jobs-table", "data"),
    prevent_initial_call=True,
)
def refresh_image(
    img_ind,
    data_project_dict,
    row,
    data_table,
):
    """
    This callback updates the image in the display
    Args:
        img_ind:            Index of image according to the slider value
        log:                Log toggle
        data_project_dict:  Selected data
        row:                Selected job (model)
        data_table:         Data in table of jobs
    Returns:
        img-output:         Output figure
    """
    start = time.time()
    # Get selected job type
    if row and len(row) > 0 and row[0] < len(data_table):
        selected_job_type = data_table[row[0]]["job_type"]
    else:
        selected_job_type = None

    if selected_job_type == "prediction_model":
        job_id = data_table[row[0]]["experiment_id"]
        data_path = pathlib.Path(f"{DATA_DIR}/mlex_store/{USER}/{job_id}")

        with open(f"{data_path}/.file_manager_vars.pkl", "rb") as file:
            data_project_dict = pickle.load(file)
    data_project = DataProject.from_dict(data_project_dict, api_key=TILED_KEY)
    if (
        len(data_project.datasets) > 0
        and data_project.datasets[-1].cumulative_data_count > 0
    ):
        fig, uri = data_project.read_datasets(indices=[img_ind], resize=True)
        fig = fig[0]
        uri = uri[0]
    else:
        uri = None
        fig = plot_figure()
    logger.info(f"Time to read data: {time.time() - start}")
    return fig, uri


@callback(
    Output("img-slider", "max", allow_duplicate=True),
    Output("img-slider", "value", allow_duplicate=True),
    Input("jobs-table", "selected_rows"),
    Input("jobs-table", "data"),
    State("img-slider", "value"),
    prevent_initial_call=True,
)
def update_slider_boundaries_prediction(
    row,
    data_table,
    slider_ind,
):
    """
    This callback updates the slider boundaries according to the selected job type
    Args:
        row:                Selected row (job)
        data_table:         Lists of jobs
        slider_ind:         Slider index
    Returns:
        img-slider:         Maximum value of the slider
        img-slider:         Slider index
    """
    # Get selected job type
    if row and len(row) > 0 and row[0] < len(data_table):
        selected_job_type = data_table[row[0]]["job_type"]
    else:
        selected_job_type = None

    # If selected job type is train_model or tune_model
    if selected_job_type == "prediction_model":
        job_id = data_table[row[0]]["experiment_id"]
        data_path = pathlib.Path(f"{DATA_DIR}/mlex_store/{USER}/{job_id}")

        with open(f"{data_path}/.file_manager_vars.pkl", "rb") as file:
            data_project_dict = pickle.load(file)
        data_project = DataProject.from_dict(data_project_dict, api_key=TILED_KEY)

        # Check if slider index is out of bounds
        if (
            len(data_project.datasets) > 0
            and slider_ind > data_project.datasets[-1].cumulative_data_count - 1
        ):
            slider_ind = 0

        return data_project.datasets[-1].cumulative_data_count - 1, slider_ind

    else:
        raise PreventUpdate


@callback(
    Output("img-slider", "max"),
    Output("img-slider", "value"),
    Input({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    Input("jobs-table", "selected_rows"),
    State("img-slider", "value"),
    prevent_initial_call=True,
)
def update_slider_boundaries_new_dataset(
    data_project_dict,
    row,
    slider_ind,
):
    """
    This callback updates the slider boundaries according to the selected job type
    Args:
        data_project_dict:  Data project dictionary
        row:                Selected row (job)
        slider_ind:         Slider index
    Returns:
        img-slider:         Maximum value of the slider
        img-slider:         Slider index
    """
    data_project = DataProject.from_dict(data_project_dict, api_key=TILED_KEY)
    if len(data_project.datasets) > 0:
        max_ind = data_project.datasets[-1].cumulative_data_count - 1
    else:
        max_ind = 0

    slider_ind = min(slider_ind, max_ind)
    return max_ind, slider_ind


@callback(
    Output("img-slider", "value", allow_duplicate=True),
    Input("img-labeled-indx", "value"),
    prevent_initial_call=True,
)
def update_slider_value(labeled_img_ind):
    """
    This callback updates the slider value according to the labeled image index
    Args:
        labeled_img_ind:    Index of labeled image
    Returns:
        img-slider:         Slider index
    """
    return labeled_img_ind


@callback(
    Output("img-label", "children"),
    Input("img-uri", "data"),
    Input("event-id", "value"),
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    prevent_initial_call=True,
)
def refresh_label(uri, event_id, data_project_dict):
    """
    This callback updates the label of the image in the display
    Args:
        uri:                URI of the image
        event_id:           Event ID
        data_project_dict:  Data project dictionary
    Returns:
        img-label:          Label of the image
    """
    data_project = DataProject.from_dict(data_project_dict, api_key=TILED_KEY)
    label = "Not labeled"
    if event_id is not None and uri is not None:
        datasets = requests.get(
            f"{SPLASH_URL}/datasets",
            params={
                "uris": uri,
                "event_id": event_id,
                "project": data_project.project_id,
            },
        ).json()
        if len(datasets) > 0:
            for dataset in datasets:
                for tag in dataset["tags"]:
                    if tag["event_id"] == event_id:
                        label = f"Label: {tag['name']}"
                        break
    return label


@callback(
    Output("results-plot", "figure"),
    Output("results-plot", "style"),
    Input("img-slider", "value"),
    Input("jobs-table", "selected_rows"),
    Input("interval", "n_intervals"),
    State("jobs-table", "data"),
    State("results-plot", "figure"),
    prevent_initial_call=True,
)
def refresh_results(img_ind, row, interval, data_table, current_fig):
    """
    This callback updates the results in the display
    Args:
        img_ind:            Index of image according to the slider value
        row:                Selected job (model)
        data_table:         Data in table of jobs
        current_fig:        Current loss plot
    Returns:
        results_plot:       Output results with probabilities per class
        results_style:      Modify visibility of output results
    """
    changed_id = dash.callback_context.triggered[-1]["prop_id"]
    results_fig = dash.no_update
    results_style_fig = dash.no_update

    if row is not None and len(row) > 0 and row[0] < len(data_table):
        # Get the job logs
        try:
            job_data = TableJob.get_job(
                USER, "mlcoach", job_id=data_table[row[0]]["job_id"]
            )
        except Exception:
            logger.error(traceback.format_exc())
            raise PreventUpdate
        log = job_data["logs"]

        # Plot classification probabilities per class
        if (
            "interval" not in changed_id
            and data_table[row[0]]["job_type"] == "prediction_model"
        ):
            job_id = data_table[row[0]]["experiment_id"]
            data_path = pathlib.Path(f"{DATA_DIR}/mlex_store/{USER}/{job_id}")

            # Check if the results file exists
            if os.path.exists(f"{data_path}/results.parquet"):
                df_prob = pd.read_parquet(f"{data_path}/results.parquet")

                # Get the probabilities for the selected image
                probs = df_prob.iloc[img_ind]
                results_fig = get_class_prob(probs)
                results_style_fig = {
                    "width": "100%",
                    "height": "100%",
                    "display": "block",
                }

        # Plot the loss plot
        elif log and data_table[row[0]]["job_type"] == "train_model":
            if data_table[row[0]]["job_type"] == "train_model":
                job_id = data_table[row[0]]["experiment_id"]
                loss_file_path = (
                    f"{DATA_DIR}/mlex_store/{USER}/{job_id}/training_log.csv"
                )
                if os.path.exists(loss_file_path):
                    results_fig = generate_loss_plot(loss_file_path)
                    results_style_fig = {
                        "width": "100%",
                        "height": "100%",
                        "display": "block",
                    }

        else:
            results_fig = []
            results_style_fig = {"display": "none"}

        # Do not update the plot unless loss plot changed
        if (
            current_fig
            and results_fig != dash.no_update
            and current_fig["data"][0]["y"] == list(results_fig["data"][0]["y"])
        ):
            results_fig = dash.no_update
            results_style_fig = dash.no_update

        return results_fig, results_style_fig
    elif current_fig:
        return [], {"display": "none"}
    else:
        raise PreventUpdate


@callback(
    Output("warning-modal", "is_open"),
    Output("warning-msg", "children"),
    Input("warning-cause", "data"),
    Input("warning-cause-execute", "data"),
    prevent_initial_call=True,
)
def open_warning_modal(warning_cause, warning_cause_exec):
    """
    This callback opens a warning/error message
    Args:
        warning_cause:      Cause that triggered the warning
        warning_cause_exec: Execution-related cause that triggered the warning
        is_open:            Close/open state of the warning
    """
    if warning_cause_exec == "no_row_selected":
        return False, "Please select a trained model from the List of Jobs."
    elif warning_cause_exec == "no_dataset":
        return False, "Please upload the dataset before submitting the job."
    else:
        return False, ""


@callback(
    Output("warning-modal", "is_open", allow_duplicate=True),
    Output("warning-msg", "children", allow_duplicate=True),
    Input("ok-button", "n_clicks"),
    prevent_initial_call=True,
)
def close_warning_modal(ok_n_clicks):
    return False, ""
