import base64, sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

import numpy as np
import pandas as pd
from PIL import Image
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_loss_plot(log, start):
    '''
    Generate loss plot
    Args:
        log:    job logs with the loss/accuracy per epoch
        start:  index where the list of loss values start
    Returns:
        loss plot
    '''
    end = log.find('Train process completed')
    if end == -1:
        end = len(log)
    log = log[start:end]
    df = pd.read_csv(StringIO(log.replace('\n\n', '\n')), sep=',')
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


def get_class_prob(log, start, filename):
    '''
    Generate plot of probabilities per class
    Args:
        log:    job logs with the list of probabilities per file
        start:  index where the list of loss values start
    Returns:
        plot of probabilities per class
    '''
    end = log.find('Prediction process completed')
    if end == -1:
        end = len(log)
    log = log[start:end]
    try:
        df = pd.read_csv(StringIO(log.replace('\n\n', '\n')), sep=',')
        res = df.loc[df['filename'] == filename]    # search results for the selected file
        if res.shape[0]>1:
            res = res.iloc[[0]]
        fig = px.bar(res.iloc[: , 1:])
        fig.update_layout(yaxis_title="probability", margin=dict(l=20, r=20, t=20, b=20))
        fig.update_xaxes(showgrid=False,
                         visible=False,
                         showticklabels=False,
                         zeroline=False)
        return fig
    except Exception as err:
        print(err)
        return go.Figure(go.Scatter(x=[], y=[]))


def plot_figure(image=None):
    '''
    Plot input data
    '''
    if not image:           # Create a blank image
        blank_image = np.zeros((300,300,3), dtype=np.uint8)
        image = Image.fromarray(blank_image)
    fig = px.imshow(image,
                    height=400)
    fig.update_xaxes(showgrid=False,
                     showticklabels=False,
                     zeroline=False)
    fig.update_yaxes(showgrid=False,
                     showticklabels=False,
                     zeroline=False)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=10))
    png = plotly.io.to_image(fig, format='jpg')
    png_base64 = base64.b64encode(png).decode('ascii')
    return "data:image/jpg;base64,{}".format(png_base64)