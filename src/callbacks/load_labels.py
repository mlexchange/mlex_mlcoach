import time
from datetime import datetime, timezone

import dash
import requests
from dash import Input, Output, State, callback
from file_manager.data_project import DataProject

from app_layout import SPLASH_URL


@callback(
    Output("event-id", "options"),
    Output("modal-load-splash", "is_open"),
    Input("button-load-splash", "n_clicks"),
    Input("confirm-load-splash", "n_clicks"),
    Input({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    State("timezone-browser", "value"),
    prevent_initial_call=True,
)
def load_from_splash_modal(
    load_n_click, confirm_load, data_project_dict, timezone_browser
):
    """
    Load labels from splash-ml associated with the project_id
    Args:
        load_n_click:       Number of clicks in load from splash-ml button
        confirm_load:       Number of clicks in confim button within loading from splash-ml modal
        data_project_dict:  Data project information
        timezone_browser:   Timezone of the browser
    Returns:
        event_id:           Available tagging event IDs associated with the current data project
        modal_load_splash:  True/False to open/close loading from splash-ml modal
    """
    changed_id = dash.callback_context.triggered[-1]["prop_id"]
    if (
        changed_id == "confirm-load-splash.n_clicks"
    ):  # if confirmed, load chosen tagging event
        return dash.no_update, False
    # If unconfirmed, retrieve the tagging event IDs associated with the current data project
    data_project = DataProject.from_dict(data_project_dict)
    if len(data_project.datasets) > 0:
        start = time.time()
        response = requests.get(
            f"{SPLASH_URL}/events", params={"page[offset]": 0, "page[limit]": 1000}
        )
        event_ids = response.json()

        # Present the tagging event options with their corresponding tagger id and runtime
        options = []
        for tagging_event in event_ids:
            tagging_event_time = datetime.strptime(
                tagging_event["run_time"], "%Y-%m-%dT%H:%M:%S.%f"
            )
            tagging_event_time = (
                tagging_event_time.replace(tzinfo=timezone.utc)
                .astimezone(tz=None)
                .strftime("%d-%m-%Y %H:%M:%S")
            )
            options.append(
                {
                    "label": f"Tagger ID: {tagging_event['tagger_id']}, \
                                    modified: {tagging_event_time}",
                    "value": tagging_event["uid"],
                }
            )
        print(f"Time taken to fetch tagging events: {time.time() - start}", flush=True)
        return options, True
    else:
        return dash.no_update, dash.no_update


@callback(
    Output("img-labeled-indx", "options"),
    Input("confirm-load-splash", "n_clicks"),
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    State("event-id", "value"),
    prevent_initial_call=True,
)
def get_labeled_indx(confirm_load, data_project_dict, event_id):
    """
    This callback retrieves the indexes of labeled images
    Args:
        confirm_load:       Number of clicks in "confirm loading from splash" button
        data_project_dict:  Data project information
        event_id:           Tagging event id for version control of tags
    Returns:
        List of indexes of labeled images in this tagging event
    """
    data_project = DataProject.from_dict(data_project_dict)
    num_imgs = data_project.datasets[-1].cumulative_data_count
    data_uris = data_project.read_datasets(list(range(num_imgs)), just_uri=True)
    options = []
    if num_imgs > 0:
        response = requests.post(
            f"{SPLASH_URL}/datasets/search",
            params={"page[limit]": num_imgs},
            json={"event_id": event_id},
        )
        for dataset in response.json():
            index = next(
                (
                    i
                    for i, uri in enumerate(data_uris)
                    if uri == dataset["uri"] and len(dataset["tags"]) > 0
                ),
                None,
            )
            if index is not None:
                options.append(index)
    return sorted(options)
