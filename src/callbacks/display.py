import traceback

from dash import Input, Output, State, callback
from file_manager.data_project import DataProject
from mlex_utils.prefect_utils.core import get_children_flow_run_ids

from src.app_layout import DATA_TILED_KEY, USER, logger
from src.utils.data_utils import hash_list_of_strings, tiled_results
from src.utils.label_utils import labels
from src.utils.mask_utils import tiled_mask
from src.utils.plot_utils import get_class_prob, plot_figure


@callback(
    Output(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "project-name-id",
            "aio_id": "mlcoach-jobs",
        },
        "data",
    ),
    Input({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    prevent_initial_call=True,
)
def update_project_name(data_project_dict):
    data_project = DataProject.from_dict(data_project_dict)
    data_uris = [dataset.uri for dataset in data_project.datasets]
    project_name = hash_list_of_strings(data_uris)
    return project_name


@callback(
    Output("mask-store", "data"),
    Input("mask-dropdown", "value"),
)
def update_mask(mask):
    """
    This callback updates the mask in the display
    Args:
        mask:               Mask to be applied to the image
    Returns:
        mask-store:         Mask to be applied to the image
    """
    return tiled_mask.get_data_by_trimmed_uri(mask) if mask != "None" else None


@callback(
    Output(
        "img-output", "src"
    ),  # TODO: Populate ("img-output-store", "data") for clientside callback
    Output("img-uri", "data"),
    Input("img-slider", "value"),
    Input({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    Input("log-transform", "value"),
    Input("min-max-percentile", "value"),
    Input("mask-store", "data"),
    prevent_initial_call=True,
)
def refresh_image(img_ind, data_project_dict, log_transform, min_max_percentile, mask):
    """
    This callback updates the image in the display
    Args:
        img_ind:            Index of image according to the slider value
        data_project_dict:  Selected data
    Returns:
        img-output:         Output figure
    """
    data_project = DataProject.from_dict(data_project_dict, api_key=DATA_TILED_KEY)
    if (
        len(data_project.datasets) > 0
        and data_project.datasets[-1].cumulative_data_count > 0
    ):
        img, uri = data_project.read_datasets(
            indices=[img_ind],
            resize=True,
            log=log_transform,
            percentiles=min_max_percentile,
            export="pillow",
        )
        fig = plot_figure(img[0])
        uri = uri[0]
    else:
        uri = None
        fig = plot_figure()
    return fig, uri


@callback(
    Output("img-slider", "max"),
    Output("img-slider", "value"),
    Input({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    State("img-slider", "value"),
    prevent_initial_call=True,
)
def update_slider_boundaries_new_dataset(
    data_project_dict,
    slider_ind,
):
    """
    This callback updates the slider boundaries
    Args:
        data_project_dict:  Data project dictionary
        slider_ind:         Slider index
    Returns:
        img-slider:         Maximum value of the slider
        img-slider:         Slider index
    """
    data_project = DataProject.from_dict(data_project_dict, api_key=DATA_TILED_KEY)
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
    return int(labeled_img_ind)


@callback(
    Output("img-label", "children"),
    Input("img-uri", "data"),
    Input("event-id", "value"),
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
def refresh_label(uri, event_id, project_name):
    """
    This callback updates the label of the image in the display
    Args:
        uri:                URI of the image
        event_id:           Event ID
        project_name:       Name of the project
    Returns:
        img-label:          Label of the image
    """
    label = "Not labeled"
    if event_id is not None and uri is not None:
        try:
            label = labels.get_label(project_name, event_id, uri)
        except Exception:
            logger.error(traceback.format_exc())
    return label


@callback(
    Output("results-plot", "figure"),
    Output("results-plot", "style"),
    Input("img-slider", "value"),
    Input("show-results", "value"),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "train-dropdown",
            "aio_id": "mlcoach-jobs",
        },
        "value",
    ),
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
def refresh_results(img_ind, show_res, job_id, project_name):
    if show_res:
        child_job_id = get_children_flow_run_ids(job_id)[1]
        expected_result_uri = f"{USER}/{project_name}/{child_job_id}/probabilities"
        probs = tiled_results.get_data_by_trimmed_uri(expected_result_uri, indx=img_ind)
        results_fig = get_class_prob(probs)
        results_style_fig = {
            "width": "100%",
            "height": "100%",
            "display": "block",
        }
    else:
        results_fig = get_class_prob()
        results_style_fig = {"display": "none"}
    return results_fig, results_style_fig


@callback(
    Output("sidebar-offcanvas", "is_open", allow_duplicate=True),
    Output("main-display", "style"),
    Input("sidebar-view", "n_clicks"),
    State("sidebar-offcanvas", "is_open"),
    prevent_initial_call=True,
)
def toggle_sidebar(n_clicks, is_open):
    if is_open:
        style = {}
    else:
        style = {"padding": "0px 10px 0px 510px", "width": "100%"}
    return not is_open, style
