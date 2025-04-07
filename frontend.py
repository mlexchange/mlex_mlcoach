import os

from dash import Input, Output

from src.app_layout import app
from src.callbacks.display import (  # noqa: F401
    refresh_label,
    update_slider_boundaries_new_dataset,
    update_slider_value,
)
from src.callbacks.execute import (  # noqa: F401
    allow_show_stats,
    run_inference,
    run_train,
    show_training_stats,
    update_model_parameters,
)
from src.callbacks.load_labels import (  # noqa: F401
    get_labeled_indices,
    refresh_tagging_events,
)
from src.callbacks.load_parameters import load_model_parameters  # noqa: F401

APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = os.getenv("APP_PORT", "8062")
READ_DIR = os.getenv("READ_DIR", "/tiled_storage")
DIR_MOUNT = os.getenv("DIR_MOUNT", READ_DIR)

server = app.server

# TODO: Readd this
# app.clientside_callback(
#     ClientsideFunction(namespace="clientside", function_name="transform_image"),
#     Output("img-output", "src"),
#     Input("log-transform", "value"),
#     Input("min-max-percentile", "value"),
#     Input("mask-store", "data"),
#     Input("img-output-store", "data"),
#     prevent_initial_call=True,
# )


app.clientside_callback(
    """
    function(n) {
        if (typeof Intl === 'object' && typeof Intl.DateTimeFormat === 'function') {
            const dtf = Intl.DateTimeFormat();
            if (typeof dtf === 'object' && typeof dtf.resolvedOptions === 'function') {
                const ro = dtf.resolvedOptions();
                if (typeof ro === 'object' && typeof ro.timeZone === 'string') {
                    return ro.timeZone;
                }
            }
        }
        return 'Timezone information not available';
    }
    """,
    Output("timezone-browser", "data"),
    Input("interval", "n_intervals"),
)


if __name__ == "__main__":
    app.run_server(debug=True, host=APP_HOST, port=APP_PORT)
