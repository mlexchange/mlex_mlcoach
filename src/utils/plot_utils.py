import base64

import dash_bootstrap_components as dbc
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from dash import html
from dash_iconify import DashIconify
from PIL import Image


def get_class_prob(probs=None):
    """
    Generate plot of probabilities per class
    Args:
        prob:L  probabilities per class
    Returns:
        plot of probabilities per class
    """
    if probs is None:
        return go.Figure()
    else:
        probs.name = None
        probs = probs.to_frame().T
        fig = px.bar(probs)
        fig.update_layout(
            yaxis_title="probability",
            legend_title_text="Labels",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig.update_xaxes(
            showgrid=False, visible=False, showticklabels=False, zeroline=False
        )
    return fig


def plot_figure(image=None):
    """
    Plot input data
    """
    if not image:  # Create a blank image
        blank_image = np.zeros((500, 500, 3), dtype=np.uint8)
        image = Image.fromarray(blank_image)
        fig = px.imshow(image, height=300, width=300)
    else:
        fig = px.imshow(image, height=300, width=300)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(margin=dict(l=0, r=10, t=0, b=10))
    png = plotly.io.to_image(fig, format="jpg")
    png_base64 = base64.b64encode(png).decode("ascii")
    return "data:image/jpg;base64,{}".format(png_base64)


def generate_notification(title, color, icon, message=""):
    iconify_icon = DashIconify(
        icon=icon,
        width=24,
        height=24,
        style={"verticalAlign": "middle"},
    )
    return [
        dbc.Toast(
            id="auto-toast",
            children=[
                html.Div(
                    [
                        iconify_icon,
                        html.Span(title, style={"margin-left": "10px"}),
                    ],
                    className="d-flex align-items-center",
                ),
                html.P(message, className="mb-0"),
            ],
            duration=4000,
            is_open=True,
            color=color,
            style={
                "position": "fixed",
                "top": 66,
                "right": 10,
                "width": 350,
                "zIndex": 9999,
            },
        )
    ]
