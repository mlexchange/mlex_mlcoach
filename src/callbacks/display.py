import os
import pathlib

import dash
import pandas as pd
import requests
from dash import Input, Output, State, callback
from dash.exceptions import PreventUpdate
from file_manager.data_project import DataProject

from app_layout import SPLASH_URL, TILED_KEY, USER
from utils.job_utils import TableJob
from utils.plot_utils import generate_loss_plot, get_class_prob, plot_figure


@callback(
    Output("img-output", "src"),
    Output("img-slider", "max"),
    Output("img-slider", "value"),
    Output("img-label", "children"),
    Output("warning-cause", "data"),
    Input({"base_id": "file-manager", "name": "docker-file-paths"}, "data"),
    Input("img-slider", "value"),
    Input("img-labeled-indx", "value"),
    Input("jobs-table", "selected_rows"),
    Input("event-id", "value"),
    Input({"base_id": "file-manager", "name": "log-toggle"}, "on"),
    State("jobs-table", "data"),
    State({"base_id": "file-manager", "name": "project-id"}, "data"),
    prevent_initial_call=True,
)
def refresh_image(
    file_paths, img_ind, labeled_img_ind, row, event_id, log, data_table, project_id
):
    """
    This callback updates the image in the display
    Args:
        file_paths:         Selected data files
        img_ind:            Index of image according to the slider value
        labeled_img_ind:    Indexes of the labeled images in this data set
        row:                Selected job (model)
        event_id:           Tagging event id for version control of tags
        log:                Log toggle
        data_table:         Data in table of jobs
        project_id:         Data project id
    Returns:
        img-output:         Output figure
        img-slider-max:     Maximum value of the slider according to the dataset (train vs test)
        img-slider-value:   Current value of the slider
        label-output:       Output label
        warning-cause:      Cause that triggered warning pop-up
    """
    changed_id = dash.callback_context.triggered[-1]["prop_id"]
    if "img-labeled-indx" in changed_id and labeled_img_ind is not None:
        img_ind = labeled_img_ind
    data_project = DataProject()
    if (
        row is not None
        and len(row) > 0
        and row[0] < len(data_table)
        and data_table[row[0]]["job_type"].split()[0] == "prediction_model"
    ):
        job_id = data_table[row[0]]["experiment_id"]
        data_path = pathlib.Path("data/mlexchange_store/{}/{}".format(USER, job_id))
        data_info = pd.read_parquet(f"{data_path}/data_info.parquet", engine="pyarrow")
        data_project.init_from_dict(data_info.to_dict("records"), api_key=TILED_KEY)
    else:
        data_project.init_from_dict(file_paths)
    if len(data_project.data) > 0:
        try:
            slider_max = len(data_project.data) - 1
            if img_ind > slider_max:
                img_ind = 0
            fig, uri = data_project.data[img_ind].read_data(log=log)
            label = "Not labeled"
            if event_id is not None:
                datasets = requests.get(
                    f"{SPLASH_URL}/datasets",
                    params={"uris": uri, "event_id": event_id, "project": project_id},
                ).json()
                if len(datasets) > 0:
                    for dataset in datasets:
                        for tag in dataset["tags"]:
                            if tag["event_id"] == event_id:
                                label = f"Label: {tag['name']}"
                                break
            return fig, slider_max, img_ind, label, dash.no_update
        except Exception as e:
            print(f"Exception in refresh_image callback {e}")
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                "wrong_dataset",
            )
    else:
        return plot_figure(), 0, 0, "", dash.no_update


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
        job_data = TableJob.get_job(
            USER, "mlcoach", job_id=data_table[row[0]]["job_id"]
        )
        log = job_data["logs"]

        # Plot classification probabilities per class
        if (
            "interval" not in changed_id
            and data_table[row[0]]["job_type"] == "prediction_model"
        ):
            job_id = data_table[row[0]]["experiment_id"]
            data_path = pathlib.Path("data/mlexchange_store/{}/{}".format(USER, job_id))

            # Check if the results file exists
            if os.path.exists(f"{data_path}/results.parquet"):
                # Load the data information
                data_project = DataProject()
                df_prob = pd.read_parquet(f"{data_path}/results.parquet")
                data_info = pd.read_parquet(
                    f"{data_path}/data_info.parquet", engine="pyarrow"
                )
                data_project.init_from_dict(
                    data_info.to_dict("records"), api_key=TILED_KEY
                )

                # Get the probabilities for the selected image
                probs = df_prob.loc[data_project.data[img_ind].uri]
                probs = probs.to_frame().T.reset_index(drop=True)
                results_fig = get_class_prob(probs)
                results_style_fig = {
                    "width": "100%",
                    "height": "100%",
                    "display": "block",
                }

        # Plot the loss plot
        elif log and data_table[row[0]]["job_type"] == "train_model":
            if data_table[row[0]]["job_type"] == "train_model":
                start = log.find("epoch")
                if start > -1 and len(log) > start + 5:
                    results_fig = generate_loss_plot(log, start)
                    results_style_fig = {
                        "width": "100%",
                        "height": "100%",
                        "display": "block",
                    }

        # Do not update the plot unless loss plot changed
        if (
            current_fig
            and results_fig != dash.no_update
            and current_fig["data"][0]["x"] == list(results_fig["data"][0]["x"])
        ):
            results_fig = dash.no_update
            results_style_fig = dash.no_update

        return results_fig, results_style_fig
    else:
        raise PreventUpdate


@callback(
    Output("warning-modal", "is_open"),
    Output("warning-msg", "children"),
    Input("warning-cause", "data"),
    Input("warning-cause-execute", "data"),
    Input("ok-button", "n_clicks"),
    State("warning-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_warning_modal(warning_cause, warning_cause_exec, ok_n_clicks, is_open):
    """
    This callback toggles a warning/error message
    Args:
        warning_cause:      Cause that triggered the warning
        warning_cause_exec: Execution-related cause that triggered the warning
        ok_n_clicks:        Close the warning
        is_open:            Close/open state of the warning
    """
    changed_id = dash.callback_context.triggered[0]["prop_id"]
    if "ok-button.n_clicks" in changed_id:
        return False, ""
    if warning_cause == "wrong_dataset":
        return False, ""
        # return not is_open, "The dataset you have selected is not supported."
    if warning_cause_exec == "no_row_selected":
        return not is_open, "Please select a trained model from the List of Jobs."
    if warning_cause_exec == "no_dataset":
        return not is_open, "Please upload the dataset before submitting the job."
    else:
        return False, ""
