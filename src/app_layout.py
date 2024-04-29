import os
import pathlib

import dash
import dash_bootstrap_components as dbc
import diskcache
from dash import html
from dash.long_callback import DiskcacheLongCallbackManager
from dotenv import load_dotenv
from file_manager.main import FileManager

from src.components.header import header
from src.components.job_table import job_table
from src.components.main_display import main_display
from src.components.resources_setup import resources_setup
from src.components.sidebar import sidebar
from src.utils.job_utils import TableJob, get_host
from src.utils.model_utils import get_model_list

load_dotenv(".env")

USER = "admin"
DATA_DIR = os.getenv("DATA_DIR")
DOCKER_DATA = pathlib.Path.home() / "data"
UPLOAD_FOLDER_ROOT = DOCKER_DATA / "upload"
SPLASH_URL = os.getenv("SPLASH_URL")
DEFAULT_TILED_URI = os.getenv("DEFAULT_TILED_URI")
TILED_KEY = os.getenv("TILED_KEY")
if TILED_KEY == "":
    TILED_KEY = None
HOST_NICKNAME = os.getenv("HOST_NICKNAME")
num_processors, num_gpus = get_host(HOST_NICKNAME)

# SETUP DASH APP
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "../assets/mlex-style.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
]
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    long_callback_manager=long_callback_manager,
)
app.title = "MLCoach"
app._favicon = "mlex.ico"
dash_file_explorer = FileManager(
    DATA_DIR,
    open_explorer=False,
    api_key=TILED_KEY,
)
dash_file_explorer.init_callbacks(app)

# DEFINE LAYOUT
app.layout = html.Div(
    [
        header("MLExchange | MLCoach", "https://github.com/mlexchange/mlex_mlcoach"),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            sidebar(
                                dash_file_explorer.file_explorer,
                                get_model_list(),
                                TableJob.get_counter(USER),
                            ),
                            width=4,
                        ),
                        dbc.Col(main_display(job_table()), width=8),
                        html.Div(id="dummy-output"),
                    ]
                ),
                resources_setup(num_processors, num_gpus),
            ],
            fluid=True,
        ),
    ]
)
