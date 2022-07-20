import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests


class SimpleJob:
    def __init__(self,
                 service_type,
                 working_directory,
                 uri,
                 cmd,
                 kwargs=None,
                 mlex_app='mlcoach'):
        self.mlex_app = mlex_app
        self.service_type = service_type
        self.working_directory = working_directory
        self.job_kwargs = {'uri': uri,
                           'type': 'docker',
                           'cmd': cmd,
                           'kwargs': kwargs}

    def submit(self, user, num_cpus, num_gpus):
        '''
        Sends job to computing service
        Args:
            user:       user UID
            num_cpus:   Number of CPUs
            num_gpus:   Number of GPUs
        Returns:
            Workflow status
        '''
        workflow = {'user_uid': user,
                    'job_list': [self.__dict__],
                    'host_list': ['mlsandbox.als.lbl.gov', 'local.als.lbl.gov', 'vaughan.als.lbl.gov'],
                    'dependencies': {'0':[]},
                    'requirements': {'num_processors': num_cpus,
                                     'num_gpus': num_gpus,
                                     'num_nodes': 1}}
        print(workflow)
        url = 'http://job-service:8080/api/v0/workflows'
        return requests.post(url, json=workflow).status_code


# Queries the job from the computing database
def get_job(user, mlex_app, job_type=None, deploy_location=None):
    url = 'http://job-service:8080/api/v0/jobs?'
    if user:
        url += ('&user=' + user)
    if mlex_app:
        url += ('&mlex_app=' + mlex_app)
    if job_type:
        url += ('&job_type=' + job_type)
    if deploy_location:
        url += ('&deploy_location=' + deploy_location)
    response = requests.get(url)
    return response.json()


def get_class_prob(log, start, filename):
    end = log.find('Prediction process completed')
    if end == -1:
        end = len(log)
    log = log[start:end]
    try:
        df = pd.read_csv(StringIO(log.replace('\n\n', '\n')), sep=' ')
        res = df.loc[df['filename'] == filename]    # search results for the selected file
        fig = px.bar(res.iloc[: , 1:])
        fig.update_layout(yaxis_title="probability")
        fig.update_xaxes(showgrid=False,
                         showticklabels=False,
                         zeroline=False)
        return fig #res.to_string(index=False)
    except Exception as err:
        return go.Figure(go.Scatter(x=[], y=[]))


# Generate loss plot
def generate_figure(log, start):
    end = log.find('Train process completed')
    if end == -1:
        end = len(log)
    log = log[start:end]
    df = pd.read_csv(StringIO(log.replace('\n\n', '\n')), sep=' ')
    try:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for col in list(df.columns)[1:]:
            if 'loss' in col:
                fig.add_trace(go.Scatter(x=df['epoch'], y=df[col], name=col), secondary_y=False)
                fig.update_yaxes(title_text="loss", secondary_y=False)
            else:
                fig.add_trace(go.Scatter(x=df['epoch'], y=df[col], name=col), secondary_y=True)
                fig.update_yaxes(title_text="accuracy", secondary_y=True, range=[0,1])
        fig.update_layout(xaxis_title="epoch", margin=dict(l=20, r=20, t=20, b=20))
        return fig
    except Exception as e:
        print(e)
        return go.Figure(go.Scatter(x=[], y=[]))


def plot_figure(image):
    fig = px.imshow(image, height=350)
    fig.update_xaxes(showgrid=False,
                     showticklabels=False,
                     zeroline=False)
    fig.update_yaxes(showgrid=False,
                     showticklabels=False,
                     zeroline=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=10))
    fig.update_traces(dict(showscale=False, coloraxis=None))
    return fig

# saves model as an .h5 file on local disk
def save_model(model, save_path='my_model.h5'):
    # save model
    model.save(save_path)
    print("Saved to disk")


def model_list_GET_call():
    """
    Get a list of algorithms from content registry
    """
    url = 'http://content-api:8000/api/v0/models'
    model_list = requests.get(url).json()
    models = []
    for item in model_list:
        if 'mlcoach' in item['application']:
            models.append({'label': item['name'], 'value': item['content_id']})
    return models


def get_model(model_uid):
    '''
    This function gets the algorithm dict from content registry
    Args:
         model_uid:     Model UID
    Returns:
        service_type:   Backend/Frontend
        content_uri:    URI
    '''
    url = 'http://content-api:8000/api/v0/contents/{}/content'.format(model_uid)
    content = requests.get(url).json()
    if 'map' in content:
        return content['service_type'], content['uri']
    return content['service_type'], content['uri']


def get_gui_components(model_uid, comp_group):
    '''
    Returns the GUI components of the corresponding model and action
    Args:
        model_uid:  Model UID
        comp_group: Action, e.g. training, testing, etc
    Returns:
        params:     List of model parameters
    '''
    url = f'http://content-api:8000/api/v0/models/{model_uid}/model/{comp_group}/gui_params'
    response = requests.get(url)
    return response.json()


def init_counter(username):
    job_list = get_job(username, 'mlcoach')
    job_types = ['train_model', 'evaluate_model', 'prediction_model', 'transfer_learning']
    counters = [-1, -1, -1, -1]
    if job_list is not None:
        for indx, job_type in enumerate(job_types):
            for job in reversed(job_list):
                last_job = job['job_kwargs']['kwargs']['job_type'].split()
                value = int(last_job[-1])
                last_job = ' '.join(last_job[0:-1])
                if last_job == job_type:
                    counters[indx] = value
                    break
    return counters
