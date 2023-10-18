from dash import dcc, Input, Output, State, callback
import dash

from app_layout import USER
from utils.job_utils import TableJob


@callback(
    Output('jobs-table', 'data'),
    Output('log-modal', 'is_open'),
    Output('log-display', 'children'),
    Output('jobs-table', 'active_cell'),

    Input('interval', 'n_intervals'),
    Input('jobs-table', 'active_cell'),
    Input('modal-close', 'n_clicks'),

    State('jobs-table', 'data'),
    prevent_initial_call=True
)
def update_table(n, active_cell, close_clicks, current_job_table):
    '''
    This callback updates the job table, loss plot, and results according to the job status in the 
    compute service.
    Args:
        n:                  Time intervals that trigger this callback
        active_cell:        Selected cell in jobs table
        close_clicks:       Close modal with close-up details of selected cell
        current_job_table:  Current job table
    Returns:
        jobs-table:         Updates the job table
        show-plot:          Shows/hides the loss plot
        loss-plot:          Updates the loss plot according to the job status (logs)
        active_cell:        Selected cell in jobs table
    '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'modal-close.n_clicks' in changed_id:
        return dash.no_update, False, dash.no_update, None
    job_list = TableJob.get_job(USER, 'mlcoach')
    data_table = []
    if job_list is not None:
        for job in job_list:
            simple_job = TableJob.compute_job_to_table_job(job)
            data_table.insert(0, simple_job.__dict__)
    is_open = dash.no_update
    log_display = dash.no_update
    if active_cell:
        row_log = active_cell["row"]
        col_log = active_cell["column_id"]
        if col_log == 'job_logs':       # show job logs
            is_open = True
            log_display = dcc.Textarea(value=data_table[row_log]["job_logs"],
                                       style={'width': '100%', 
                                              'height': '30rem', 
                                              'font-family':'monospace'})
        if col_log == 'parameters':     # show job parameters
            is_open = True
            log_display = dcc.Textarea(value=str(job['job_kwargs']['kwargs']['params']),
                                       style={'width': '100%', 
                                              'height': '30rem', 
                                              'font-family': 'monospace'})
    if data_table == current_job_table:
        data_table = dash.no_update
    return data_table, is_open, log_display, None


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