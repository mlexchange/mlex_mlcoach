from dash import dcc, Input, Output, State, callback
import dash

from app_layout import USER
from utils.job_utils import TableJob


@callback(
    Output('jobs-table', 'data'),
    Input('interval', 'n_intervals'),
    State('jobs-table', 'data'),
)
def update_table(n, current_job_table):
    '''
    This callback updates the job table
    Args:
        n:                  Time intervals that trigger this callback
        current_job_table:  Current job table
    Returns:
        jobs-table:         Updates the job table
    '''
    job_list = TableJob.get_job(USER, 'mlcoach')
    data_table = []
    if job_list is not None:
        for job in job_list:
            simple_job = TableJob.compute_job_to_table_job(job)
            data_table.insert(0, simple_job.__dict__)
    if data_table == current_job_table:
        data_table = dash.no_update
    return data_table


@callback(
    Output('info-modal', 'is_open'),
    Output('info-display', 'children'),
    Input('show-info', 'n_clicks'),
    Input('modal-close', 'n_clicks'),
    State('jobs-table', 'data'),
    State('info-modal', 'is_open'),
    State('jobs-table', 'selected_rows'),
)
def open_job_modal(n_clicks, close_clicks, current_job_table, is_open, rows):
    '''
    This callback updates shows the job logs and parameters
    Args:
        n_clicks:           Number of clicks in "show details" button
        close_clicks:       Close modal with close-up details of selected cell
        current_job_table:  Current job table
        is_open:            Open/close modal state
        rows:               Selected rows in jobs table
    Returns:
        info-modal:         Open/closes the modal
        info-display:       Display the job logs and parameters
    '''
    if not is_open and rows is not None and len(rows) > 0:
        job_id = current_job_table[rows[0]]['job_id']
        job_data = TableJob.get_job(USER, 'mlcoach', job_id=job_id)
        logs = job_data['logs']
        params = job_data['job_kwargs']['kwargs']['params']
        info_display = dcc.Textarea(
            value= f'Parameters: {params}\n\nLogs: {logs}',
            style={
                'width': '100%', 
                'height': '30rem', 
                'font-family':'monospace'
                }
            )
        return True, info_display
    else:
        return False, dash.no_update


@callback(
    Output('jobs-table', 'selected_rows'),
    Input('deselect-row', 'n_clicks'),
    prevent_initial_call=True
)
def deselect_row(n_click):
    '''
    This callback deselects the row in the data table
    '''
    return []


@callback(
    Output('delete-modal', 'is_open'),

    Input('confirm-delete-row', 'n_clicks'),
    Input('delete-row', 'n_clicks'),
    Input('stop-row', 'n_clicks'),

    State('jobs-table', 'selected_rows'),
    State('jobs-table', 'data'),
    prevent_initial_call=True
)
def delete_row(confirm_delete, delete, stop, row, job_data):
    '''
    This callback deletes the selected model in the table
    Args:
        confirm_delete:     Number of clicks in "confirm delete row" button
        delete:             Number of clicks in "delete row" button
        stop:               Number of clicks in "stop job at row" button
        row:                Selected row in jobs table
        job_data:           Data within jobs table
    Returns:
        Open/closes confirmation modal
    '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'delete-row.n_clicks' == changed_id:
        return True
    elif 'stop-row.n_clicks' == changed_id:
        job_uid = job_data[row[0]]['job_id']
        TableJob.terminate_job(job_uid)
        return False
    else:
        job_uid = job_data[row[0]]['job_id']
        TableJob.delete_job(job_uid)
        return False