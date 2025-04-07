import json
import logging
import os

import dash
import dash_bootstrap_components as dbc
import diskcache
from dash import dcc, html
from dash.long_callback import DiskcacheLongCallbackManager
from dotenv import load_dotenv
from file_manager.main import FileManager
from mlex_utils.dash_utils.mlex_components import MLExComponents

from src.components.header import header
from src.components.infrastructure import create_infra_state_affix
from src.components.main_display import main_display
from src.components.sidebar import sidebar
from src.components.training_stats import training_stats_plot
from src.utils.model_utils import Models

load_dotenv(".env")

USER = os.getenv("USER", "mlcoach_user")
READ_DIR = os.getenv("READ_DIR", "data")
DEFAULT_TILED_URI = os.getenv("DEFAULT_TILED_URI")
DEFAULT_TILED_SUB_URI = os.getenv("DEFAULT_TILED_SUB_URI")
DATA_TILED_KEY = os.getenv("DATA_TILED_KEY")
if DATA_TILED_KEY == "":
    DATA_TILED_KEY = None
MODELFILE_PATH = os.getenv("MODELFILE_PATH", "./examples/assets/models.json")
MODE = os.getenv("MODE", "dev")
PREFECT_TAGS = json.loads(os.getenv("PREFECT_TAGS", '["mlcoach"]'))

# SETUP LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    READ_DIR,
    open_explorer=False,
    api_key=DATA_TILED_KEY,
    logger=logger,
)
dash_file_explorer.init_callbacks(app)
file_explorer = dash_file_explorer.file_explorer

# GET MODELS
models = Models(
    modelfile_path="./src/assets/default_models.json",
    model_type="classification",
)

# SETUP MLEx COMPONENTS
mlex_components = MLExComponents("dbc")
job_manager = mlex_components.get_job_manager(
    model_list=models.modelname_list,
    mode=MODE,
    aio_id="mlcoach-jobs",
    prefect_tags=PREFECT_TAGS,
)

# DEFINE LAYOUT
app.layout = html.Div(
    [
        header("MLExchange | MLCoach", "https://github.com/mlexchange/mlex_mlcoach"),
        dbc.Container(
            [
                sidebar(file_explorer, job_manager),
                main_display(training_stats_plot()),
                html.Div(id="dummy-output"),
                dcc.Store(id="timezone-browser", data=None),
                dcc.Store(id="mask-store", data=""),
                create_infra_state_affix(),
            ],
            fluid=True,
        ),
    ]
)
