from dash import Input, Output, State, callback
from file_manager.data_project import DataProject

from src.app_layout import DATA_TILED_KEY
from src.utils.label_utils import labels


@callback(
    Output("event-id", "options"),
    Input("refresh-label-events", "n_clicks"),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "project-name-id",
            "aio_id": "mlcoach-jobs",
        },
        "data",
    ),
    prevent_initial_call=True,
)
def refresh_tagging_events(refresh_n_clicks, project_name):
    return labels.get_labeling_events(project_name)


@callback(
    Output("img-labeled-indx", "options"),
    Input("event-id", "value"),
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "project-name-id",
            "aio_id": "mlcoach-jobs",
        },
        "data",
    ),
    prevent_initial_call=True,
)
def get_labeled_indices(event_id, data_project_dict, project_name):
    """
    This callback retrieves the indexes of labeled images
    Args:
        event_id:           Tagging event id for version control of tags
        data_project_dict:  Data project information
        project_name:       Project name
    Returns:
        List of indexes of labeled images in this tagging event
    """
    data_project = DataProject.from_dict(data_project_dict, api_key=DATA_TILED_KEY)
    num_imgs = data_project.datasets[-1].cumulative_data_count
    data_uris = data_project.read_datasets(list(range(num_imgs)), just_uri=True)
    indices = labels.get_labeled_indices(data_uris, project_name, event_id)
    return indices
