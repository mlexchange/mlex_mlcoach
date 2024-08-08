from dash import Input, Output, State, callback


@callback(
    Output("resources-setup", "is_open"),
    Output("warning-cause-execute", "data"),
    Input("execute", "n_clicks"),
    State("action", "value"),
    State("jobs-table", "data"),
    State("jobs-table", "selected_rows"),
    State({"base_id": "file-manager", "name": "total-num-data-points"}, "data"),
    prevent_initial_call=True,
)
def execute(execute, action_selection, job_data, row, num_data_points):
    """
    This callback submits a job request to the compute service according to the selected action & model
    Args:
        execute:            Execute button
        action_selection:   Action selected
        job_data:           Lists of jobs
        row:                Selected row (job)
        num_data_points:    Total number of data points in the dataset
    Returns:
        open/close the resources setup modal, and submits the training/prediction job accordingly
        warning_cause:      Activates a warning pop-up window if needed
    """
    if num_data_points == 0:
        return False, "no_dataset"
    elif action_selection != "train_model" and not row:
        return False, "no_row_selected"
    elif (
        action_selection != "train_model"
        and job_data[row[0]]["job_type"] != "train_model"
    ):
        return False, "no_row_selected"
    else:
        return True, ""


@callback(
    Output("resources-setup", "is_open", allow_duplicate=True),
    Output("warning-cause", "data", allow_duplicate=True),
    Input("submit", "n_clicks"),
    prevent_initial_call=True,
)
def close_resources_popup(submit):
    return False, ""
