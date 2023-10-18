from datetime import datetime, timezone

from dash import Input, Output, State, callback
import dash
import requests

from file_manager.data_project import DataProject
from app_layout import SPLASH_URL


@callback(
    Output('event-id', 'options'),
    Output('modal-load-splash', 'is_open'),

    Input('button-load-splash', 'n_clicks'),
    Input('confirm-load-splash', 'n_clicks'),
    Input({'base_id': 'file-manager', 'name': 'project-id'}, 'data'),

    State({'base_id': 'file-manager', 'name': 'docker-file-paths'}, 'data'),
    prevent_initial_call=True
)
def load_from_splash_modal(load_n_click, confirm_load, project_id, file_paths):
    '''
    Load labels from splash-ml associated with the project_id
    Args:
        load_n_click:       Number of clicks in load from splash-ml button
        confirm_load:       Number of clicks in confim button within loading from splash-ml modal
        project_id:         Data project id
        file_paths:         Data project information
    Returns:
        event_id:           Available tagging event IDs associated with the current data project
        modal_load_splash:  True/False to open/close loading from splash-ml modal
    '''
    changed_id = dash.callback_context.triggered[-1]['prop_id']
    if changed_id == 'confirm-load-splash.n_clicks':    # if confirmed, load chosen tagging event
        return dash.no_update, False
    # If unconfirmed, retrieve the tagging event IDs associated with the current data project
    data_project = DataProject()
    data_project.init_from_dict(file_paths)
    num_imgs = len(data_project.data)
    response = requests.post(f'{SPLASH_URL}/datasets/search', 
                             params={'limit': num_imgs}, 
                             json={'project': project_id})
    event_ids = []
    for dataset in response.json():
        for tag in dataset['tags']:
            if tag['event_id'] not in event_ids:
                event_ids.append(tag['event_id'])
    # Present the tagging event options with their corresponding tagger id and runtime
    options = []
    for event_id in event_ids:
        tagging_event = requests.get(f'{SPLASH_URL}/events/{event_id}').json()
        tagging_event_time = datetime.strptime(tagging_event['run_time'], "%Y-%m-%dT%H:%M:%S.%f")
        tagging_event_time = tagging_event_time.replace(tzinfo=timezone.utc).astimezone(tz=None)\
            .strftime("%d-%m-%Y %H:%M:%S")
        options.append({'label': f"Tagger ID: {tagging_event['tagger_id']}, \
                                 modified: {tagging_event_time}",
                        'value' : event_id})
    return options, True


@callback(
    Output('img-labeled-indx', 'options'),

    Input('confirm-load-splash', 'n_clicks'),

    State({'base_id': 'file-manager', 'name': 'docker-file-paths'}, 'data'),
    State({'base_id': 'file-manager', 'name': 'project-id'}, 'data'),
    State('event-id', 'value'),
    prevent_initial_call=True
)
def get_labeled_indx(confirm_load, file_paths, project_id, event_id):
    '''
    This callback retrieves the indexes of labeled images
    Args:
        confirm_load:       Number of clicks in "confirm loading from splash" button
        file_paths:         Data project information
        project_id:         Data project id
        event_id:           Tagging event id for version control of tags
    Returns:
        List of indexes of labeled images in this tagging event
    '''
    data_project = DataProject()
    data_project.init_from_dict(file_paths)
    num_imgs = len(data_project.data)
    options = []
    if num_imgs>0:
        response = requests.post(f'{SPLASH_URL}/datasets/search', 
                                 params={'limit': num_imgs}, 
                                 json={'project': project_id,
                                       'event_id': event_id})
        for dataset in response.json():
            index = next((i for i, data_obj in enumerate(data_project.data) 
                          if data_obj.uri == dataset['uri']), None)
            if index is not None:
                options.append(index)
    return options