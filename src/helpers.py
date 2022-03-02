import urllib.request

import json
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests


class SimpleJob:
    def __init__(self,
                 user,
                 job_type,
                 description,
                 deploy_location,
                 gpu,
                 data_uri,
                 container_uri,
                 container_cmd,
                 container_kwargs,
                 mlex_app = 'mlcoach'):
        self.user = user
        self.mlex_app = mlex_app
        self.job_type = job_type
        self.description = description
        self.deploy_location = deploy_location
        self.gpu = gpu
        self.data_uri = data_uri
        self.container_uri = container_uri
        self.container_cmd = container_cmd
        self.container_kwargs = container_kwargs

    def launch_job(self):
        """
        Send job to computing service
        :return:
        """
        url = 'http://job-service:8080/api/v0/jobs'
        return requests.post(url, json=self.__dict__).status_code


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
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    return data


def get_class_prob(log, start, filename):
    end = log.find('Prediction process completed')
    if end == -1:
        end = len(log)
    log = log[start:end]
    df = pd.read_csv(StringIO(log.replace('\n\n', '\n')), sep=' ')
    try:
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
    df.set_index('epoch', inplace=True)
    try:
        fig = px.line(df, markers=True)
        fig.update_layout(xaxis_title="epoch", yaxis_title="loss", margin=dict(l=20, r=20, t=20, b=20))
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
    Get the whole model registry data from the fastapi url.
    """
    url = 'http://model-api:8000/api/v0/model-list'  # current host, could be inside the docker
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    return data
