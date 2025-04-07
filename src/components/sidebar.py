import dash_bootstrap_components as dbc
from dash import dcc, html
from dash_iconify import DashIconify
from mlex_utils.dash_utils.components_bootstrap.component_utils import (
    DbcControlItem as ControlItem,
)

from src.utils.mask_utils import get_mask_options


def sidebar(file_explorer, job_manager):
    """
    Creates the dash components in the left sidebar of the app
    Args:
        file_explorer:      Dash file explorer
        job_manager:        Job manager object
    Returns:
        sidebar:            Dash sidebar
    """
    sidebar = html.Div(
        [
            dbc.Offcanvas(
                id="sidebar-offcanvas",
                is_open=True,
                backdrop=False,
                scrollable=True,
                style={
                    "padding": "80px 0px 0px 0px",
                    "width": "500px",
                },  # Avoids being covered by the navbar
                title="Controls",
                children=dbc.Accordion(
                    id="sidebar",
                    always_open=True,
                    children=[
                        dbc.AccordionItem(
                            title="Data selection",
                            children=[
                                file_explorer,
                                html.P(),
                                ControlItem(
                                    "Load Labels",
                                    "load-labels-title",
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Select(
                                                        id="event-id",
                                                        options=[],
                                                        value=None,
                                                    ),
                                                    width=10,
                                                ),
                                                dbc.Col(
                                                    dbc.Button(
                                                        DashIconify(
                                                            icon="mdi:refresh-circle",
                                                            width=20,
                                                            style={"display": "block"},
                                                        ),
                                                        id="refresh-label-events",
                                                        color="secondary",
                                                        size="sm",
                                                        className="rounded-circle",
                                                        style={
                                                            "aspectRatio": "1 / 1",
                                                            "paddingLeft": "1px",
                                                            "paddingRight": "1px",
                                                            "paddingTop": "1px",
                                                            "paddingBottom": "1px",
                                                        },
                                                    ),
                                                    className="d-flex justify-content-center align-items-center",
                                                    width=2,
                                                ),
                                            ],
                                            className="g-1",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dbc.AccordionItem(
                            title="Data transformation",
                            children=[
                                ControlItem(
                                    "",
                                    "empty-title-log-transform",
                                    dbc.Switch(
                                        id="log-transform",
                                        value=False,
                                        label="Log Transform",
                                    ),
                                ),
                                html.P(),
                                ControlItem(
                                    "Min-Max Percentile",
                                    "min-max-percentile-title",
                                    dcc.RangeSlider(
                                        id="min-max-percentile",
                                        min=0,
                                        max=100,
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                        },
                                        value=[0, 100],
                                    ),
                                ),
                                html.P(),
                                ControlItem(
                                    "Mask Selection",
                                    "mask-dropdown-title",
                                    dbc.Select(
                                        id="mask-dropdown",
                                        options=get_mask_options(),
                                        value="None",
                                    ),
                                ),
                            ],
                        ),
                        dbc.AccordionItem(
                            children=[
                                job_manager,
                                ControlItem(
                                    "",
                                    "empty-title-recons",
                                    dbc.Switch(
                                        id="show-results",
                                        value=False,
                                        label="Show Results",
                                        disabled=True,
                                    ),
                                ),
                            ],
                            title="Model Configuration",
                        ),
                    ],
                    style={"overflow-y": "scroll", "height": "90vh"},
                ),
            ),
            create_show_sidebar_affix(),
        ]
    )
    return sidebar


def create_show_sidebar_affix():
    return html.Div(
        [
            dbc.Button(
                DashIconify(icon="circum:settings", width=20),
                id="sidebar-view",
                size="sm",
                color="secondary",
                className="rounded-circle",
                style={"aspectRatio": "1 / 1"},
            ),
            dbc.Tooltip(
                "Toggle sidebar",
                target="sidebar-view",
                placement="top",
            ),
        ],
        style={
            "position": "fixed",
            "bottom": "60px",
            "right": "10px",
            "zIndex": 9999,  # Note: zIndex is unitless
            "opacity": "0.8",
        },
    )
