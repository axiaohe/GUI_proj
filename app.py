import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import threading
import time
import queue

from models import GaussianModel, RezendeModel, RosenbrockModel, ELBOModel
from stochastic_optimizers import Adam, Adamax, RMSprop, SGD
from variational_distributions.normalizing_flow import (
    NormalizingFlowVariational,
    PlanarLayer,
    Tanh,
    LeakyRelu,
    FullRankNormalLayer,
    MeanFieldNormalLayer,
)

# create a lock
lock = threading.Lock()

# Define global variables
queue_size = 400  # Maximum size of the image queue
# Queue to store generated images
output_figure_queue = queue.Queue(maxsize=queue_size)
para_elbo_figure_queue = queue.Queue(
    maxsize=queue_size
)  # Queue to store generated images
last_output_figure = go.Figure()  # Global variable to store the last figure
last_para_elbo_figure = go.Figure()  # Global variable to store the last figure

extra_layers = []
extra_layers_info = []

image_thread = None
stop_event = threading.Event()  # Event to stop the image generation thread
reset_flag = False
init_figure_flag = True
init_finished = False

app = dash.Dash(__name__)

app.layout = html.Div(
    style={
        "backgroundColor": "#f0f4f8",
        "height": "100%",
        "width": "100%",
        "margin": "0",
        "padding": "0",
        "fontFamily": "Arial, sans-serif",
    },
    children=[
        html.H1(
            "Variational Inference Visualization",
            style={"textAlign": "center", "color": "#333", "paddingTop": "20px"},
        ),
        html.Div(
            [
                dcc.Interval(
                    id="interval-component", interval=250, n_intervals=0, disabled=True
                ),
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between"},
                    children=[
                        html.Div(
                            style={"flex": "0 0 22%", "padding": "20px"},
                            children=[
                                html.Div(
                                    [
                                        html.Label(
                                            "Model Type:",
                                            style={
                                                "margin-right": "10px",
                                                "color": "#333",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="model-type",
                                            options=[
                                                {
                                                    "label": "Gaussian Model",
                                                    "value": "GaussianModel",
                                                },
                                                {
                                                    "label": "RezendeModel",
                                                    "value": "RezendeModel",
                                                },
                                                {
                                                    "label": "RosenbrockModel",
                                                    "value": "RosenbrockModel",
                                                },
                                            ],
                                            value="GaussianModel",
                                        ),
                                    ],
                                    style={"margin-bottom": "10px"},
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Optimizer Type:",
                                            style={
                                                "margin-right": "10px",
                                                "color": "#333",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="optimizer-type",
                                            options=[
                                                {"label": "Adam", "value": "Adam"},
                                                {"label": "Adamax", "value": "Adamax"},
                                                {
                                                    "label": "RMSprop",
                                                    "value": "RMSprop",
                                                },
                                                {"label": "SGD", "value": "SGD"},
                                            ],
                                            value="Adam",
                                        ),
                                    ],
                                    style={"margin-bottom": "10px"},
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Layer:",
                                            style={
                                                "margin-right": "10px",
                                                "color": "#333",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="layer-type",
                                            options=[
                                                {
                                                    "label": "PlanarLayer",
                                                    "value": "PlanarLayer",
                                                },
                                                {
                                                    "label": "FullRankNormalLayer",
                                                    "value": "FullRankNormalLayer",
                                                },
                                                {
                                                    "label": "MeanFieldNormalLayer",
                                                    "value": "MeanFieldNormalLayer",
                                                },
                                            ],
                                            value="PlanarLayer",
                                        ),
                                    ],
                                    style={"margin-bottom": "10px"},
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Activation Function:",
                                            style={
                                                "margin-right": "10px",
                                                "color": "#333",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="activation_function-type",
                                            options=[
                                                {"label": "Tanh", "value": "Tanh"},
                                                {
                                                    "label": "LeakyRelu",
                                                    "value": "LeakyRelu",
                                                },
                                                {"label": "None", "value": "None"},
                                            ],
                                            value="Tanh",
                                        ),
                                    ],
                                    style={"margin-bottom": "10px"},
                                ),
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "justifyContent": "space-between",
                                        "alignItems": "center",
                                        "marginBottom": "10px",
                                    },
                                    children=[
                                        html.Button(
                                            "Add",
                                            id="add-button",
                                            n_clicks=0,
                                            className="button button-add",
                                            style={"margin-right": "0px"},
                                        ),
                                        html.Button(
                                            "Remove",
                                            id="remove-button",
                                            n_clicks=0,
                                            className="button button-remove",
                                            style={"margin-right": "0px"},
                                        ),
                                        dcc.Input(
                                            id="n_th-layer",
                                            type="number",
                                            value=None,
                                            step=1,
                                            style={
                                                "width": "30px",
                                                "marginRight": "5px",
                                                "marginLeft": "10px",
                                                "padding": "5px",
                                                "borderRadius": "5px",
                                                "border": "1px solid #ced4da",
                                            },
                                        ),
                                        html.Label(
                                            "-th layer",
                                            style={
                                                "margin-bottom": "0",
                                                "color": "#333",
                                                "width": "60px",
                                            },
                                        ),
                                    ],
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Extra layers information: ",
                                            style={
                                                "margin-right": "10px",
                                                "color": "#333",
                                            },
                                        ),
                                        dcc.Textarea(
                                            id="layers-info",
                                            value=" ",
                                            readOnly=True,
                                            style={
                                                "width": "95%",
                                                "height": "80px",
                                                "padding": "10px",
                                                "borderRadius": "5px",
                                                "border": "1px solid #ced4da",
                                            },
                                        ),
                                    ],
                                    style={"margin-top": "5px"},
                                ),
                            ],
                        ),
                        html.Div(
                            style={"flex": "0 0 17%", "padding": "20px"},
                            children=[
                                html.Div(
                                    [
                                        html.Label(
                                            "Batch Size:",
                                            style={
                                                "margin-right": "10px",
                                                "color": "#333",
                                            },
                                        ),
                                        dcc.Input(
                                            id="batch-size",
                                            type="number",
                                            value=8,
                                            step=1,
                                            style={
                                                "width": "100%",
                                                "padding": "10px",
                                                "borderRadius": "5px",
                                                "border": "1px solid #ced4da",
                                            },
                                        ),
                                    ],
                                    style={"margin-bottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Learning Rate:",
                                            style={
                                                "margin-right": "10px",
                                                "color": "#333",
                                            },
                                        ),
                                        dcc.Input(
                                            id="learning-rate",
                                            type="number",
                                            value=1e-3,
                                            step=0.001,
                                            style={
                                                "width": "100%",
                                                "padding": "10px",
                                                "borderRadius": "5px",
                                                "border": "1px solid #ced4da",
                                            },
                                        ),
                                    ],
                                    style={"margin-bottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Max Iterations:",
                                            style={
                                                "margin-right": "10px",
                                                "color": "#333",
                                            },
                                        ),
                                        dcc.Input(
                                            id="max-iter",
                                            type="number",
                                            value=2000,
                                            step=1000,
                                            style={
                                                "width": "100%",
                                                "padding": "10px",
                                                "borderRadius": "5px",
                                                "border": "1px solid #ced4da",
                                            },
                                        ),
                                    ],
                                    style={"margin-bottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Random Seed:",
                                            style={
                                                "margin-right": "10px",
                                                "color": "#333",
                                            },
                                        ),
                                        dcc.Input(
                                            id="random-seed",
                                            type="number",
                                            value=2,
                                            step=1,
                                            style={
                                                "width": "100%",
                                                "padding": "10px",
                                                "borderRadius": "5px",
                                                "border": "1px solid #ced4da",
                                            },
                                        ),
                                    ],
                                    style={"margin-bottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Update Rate:",
                                            style={
                                                "margin-right": "10px",
                                                "color": "#333",
                                            },
                                        ),
                                        dcc.Input(
                                            id="update-rate",
                                            type="number",
                                            value=10,
                                            step=5,
                                            style={
                                                "width": "100%",
                                                "padding": "10px",
                                                "borderRadius": "5px",
                                                "border": "1px solid #ced4da",
                                            },
                                        ),
                                    ],
                                    style={"margin-bottom": "8px"},
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Initializing, please wait!",
                                            id="init_status-label",
                                            style={
                                                "color": "#007bff",
                                                "marginTop": "20px",
                                            },
                                        ),
                                    ],
                                    style={"margin-top": "20px"},
                                ),
                                html.Div(
                                    [
                                        html.Button(
                                            "Start",
                                            id="start-button",
                                            n_clicks=0,
                                            className="button button-start",
                                            style={"margin-right": "10px"},
                                        ),
                                        html.Button(
                                            "Stop",
                                            id="stop-button",
                                            n_clicks=0,
                                            className="button button-stop",
                                            style={"margin-right": "10px"},
                                        ),
                                        html.Button(
                                            "Reset",
                                            id="reset-button",
                                            n_clicks=0,
                                            className="button button-reset",
                                            style={"margin-right": "0px"},
                                        ),
                                    ],
                                    style={
                                        "margin-bottom": "20px",
                                        "margin-top": "5px",
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            style={"flex": "0 0 58%", "padding": "20px"},
                            children=[dcc.Graph(id="para_elbo-graph")],
                        ),
                    ],
                ),
                html.Div(dcc.Graph(id="output-graph")),
            ],
            style={"margin": "0 auto", "width": "90%", "maxWidth": "1200px"},
        ),
    ],
)


def image_generator(
    model_type,
    optimizer_type,
    batch_size,
    learning_rate,
    max_iter,
    random_seed,
    update_rate,
):
    global output_figure_queue, para_elbo_figure_queue, extra_layers, stop_event, reset_flag, init_figure_flag, init_finished

    np.random.seed(random_seed)
    num_dim = 2

    layers = [
        MeanFieldNormalLayer(num_dim),
        PlanarLayer(num_dim, Tanh()),
        FullRankNormalLayer(num_dim),
        PlanarLayer(num_dim, LeakyRelu()),
    ]
    layers += [PlanarLayer(num_dim, Tanh()) for _ in range(12)]
    layers.extend(extra_layers)

    variational_distribution = NormalizingFlowVariational(layers=layers)

    if model_type == "GaussianModel":
        model = GaussianModel()
    elif model_type == "RezendeModel":
        model = RezendeModel()
    else:
        model = RosenbrockModel()

    elbo_model = ELBOModel(
        model=model,
        variational_distribution=variational_distribution,
        n_samples_per_iter=batch_size,
    )

    if optimizer_type == "Adam":
        optimizer = Adam(learning_rate=learning_rate, optimization_type="max")
    elif optimizer_type == "Adamax":
        optimizer = Adamax(learning_rate=learning_rate, optimization_type="max")
    elif optimizer_type == "RMSprop":
        optimizer = RMSprop(learning_rate=learning_rate, optimization_type="max")
    else:
        optimizer = SGD(learning_rate=learning_rate, optimization_type="max")

    variational_parameters = (
        variational_distribution.initialize_variational_parameters()
    )

    iteration_count = 0
    elbo = 0
    elbo_list = []
    variational_parameters_list = []
    while iteration_count <= max_iter:
        if (not stop_event.is_set()) or init_figure_flag:
            if not init_figure_flag:
                elbo, elbo_gradient = elbo_model.evaluate_and_gradient(
                    variational_parameters
                )
                variational_parameters = optimizer.step(
                    variational_parameters, elbo_gradient
                )

                elbo_list.append(elbo)
                # variational_parameters_list.append(variational_parameters[0].item())
                variational_parameters_list.append(np.array(variational_parameters))

            if iteration_count % update_rate == 0:
                xlin = np.linspace(-2, 2, 100)
                ylin = np.linspace(-2, 3, 100)
                X, Y = np.meshgrid(xlin, ylin)
                positions = np.vstack([X.ravel(), Y.ravel()]).T
                samples = variational_distribution.draw(
                    variational_parameters, n_draws=100_000
                )

                true_pdf = np.exp(model.evaluate(positions))

                fig_output = make_subplots(
                    rows=1, cols=2, subplot_titles=("Normalizing Flow PDF", "True PDF")
                )

                fig_output.add_trace(
                    go.Histogram2d(
                        x=samples[:, 0].flatten(),
                        y=samples[:, 1].flatten(),
                        autobinx=False,
                        autobiny=False,
                        xbins=dict(start=-2, end=2, size=0.04),
                        ybins=dict(start=-2, end=3, size=0.05),
                        colorscale="Viridis",
                        colorbar=dict(title="Density", x=0.4),
                    ),
                    row=1,
                    col=1,
                )

                fig_output.add_trace(
                    go.Contour(
                        x=positions[:, 0],
                        y=positions[:, 1],
                        z=true_pdf,
                        contours=dict(
                            start=0,
                            end=np.max(true_pdf),
                            size=0.1 * np.max(true_pdf),
                            coloring="heatmap",
                        ),
                        line=dict(smoothing=0.85, color="white"),
                        colorscale="Viridis",
                        colorbar=dict(title="PDF", x=1.00),
                    ),
                    row=1,
                    col=2,
                )

                fig_output.update_layout(
                    title="Variational Inference Visualization --- Iteration: "
                    + str(0 if init_figure_flag else iteration_count),
                    title_x=0.5,
                    title_y=0.99,
                    height=600,
                    width=1200,
                    margin=dict(l=0, r=0, t=50, b=175),
                    autosize=False,
                    xaxis_title="X Axis",
                    yaxis_title="Y Axis",
                    xaxis2_title="X Axis",
                    yaxis2_title="Y Axis",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    xaxis=dict(domain=[0, 0.4]),
                    yaxis=dict(domain=[0, 1]),
                    xaxis2=dict(domain=[0.6, 1]),
                    yaxis2=dict(domain=[0, 1]),
                )

                fig_para_elbo = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=(
                        "Variational Parameters",
                        "Elbo --- Absolute Value: {:.2f}".format(np.abs(elbo)),
                    ),
                )
                
                y_value = np.array(variational_parameters_list).T
                if y_value.shape[0] == 0:
                    i_range = range(1)
                else:
                    i_range = range(y_value.shape[0])
                for i in i_range:
                    fig_para_elbo.add_trace(
                        go.Scatter(
                            x= None if init_figure_flag else np.arange(len(variational_parameters_list)),
                            y= None if init_figure_flag else y_value[i, :],
                            mode='lines',
                            line=dict(width=2),
                        ),
                        row=1,
                        col=1,
                    )

                fig_para_elbo.add_trace(
                    go.Scatter(
                        x=None if init_figure_flag else np.arange(len(elbo_list)),
                        y=None if init_figure_flag else np.array(elbo_list),
                        mode="lines",
                        line=dict(color="blue", width=2),
                    ),
                    row=1,
                    col=2,
                )

                yaxis2_range = [-25, 2]
                if elbo_list:
                    yaxis2_range = [np.min(elbo_list), np.max(elbo_list)]

                fig_para_elbo.update_layout(
                    title="Variational Parameters and Elbo --- Iteration: "
                    + str(0 if init_figure_flag else iteration_count),
                    title_x=0.5,
                    title_y=0.95,
                    height=400,
                    width=700,
                    margin=dict(l=50, r=0, t=75, b=0),
                    autosize=False,
                    xaxis_title="Iterataion",
                    yaxis_title="Parameters Value",
                    xaxis2_title="Iteration",
                    yaxis2_title="Elbo Value",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    # xaxis_range=[
                    #     0,
                    #     80 if init_figure_flag else len(variational_parameters) + 1,
                    # ],
                    # xaxis_range=[0, iteration_count],
                    xaxis_range=[0, max_iter],
                    yaxis_range=[-2, 2],
                    xaxis2_range=[0, iteration_count],
                    # xaxis2_range=[0, max_iter],
                    yaxis2_range=yaxis2_range,
                )

                fig_para_elbo.update_xaxes(
                    showgrid=False,
                    gridwidth=1,
                    gridcolor="LightGray",
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor="LightGray",
                )
                fig_para_elbo.update_yaxes(
                    showgrid=False,
                    gridwidth=1,
                    gridcolor="LightGray",
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor="LightGray",
                )

                if output_figure_queue.qsize() < queue_size - 1:
                    output_figure_queue.put(fig_output)
                    para_elbo_figure_queue.put(fig_para_elbo)
                else:
                    output_figure_queue.put(fig_output)
                    para_elbo_figure_queue.put(fig_para_elbo)
                    time.sleep(0.5)

            # start faster, prevent the first iteration from being skipped
            if init_figure_flag:
                init_finished = False
            elbo_model.evaluate_and_gradient(variational_parameters)
            if init_figure_flag:
                init_finished = True

            while True:
                if init_figure_flag and not reset_flag:
                    iteration_count = 0
                    time.sleep(0.1)
                else:
                    break
                
            iteration_count += 1
            
        else:
            if reset_flag:
                break
            else:
                time.sleep(0.1)
                pass


@app.callback(
    Output("interval-component", "disabled"),
    [
        Input("start-button", "n_clicks"),
        Input("stop-button", "n_clicks"),
        Input("reset-button", "n_clicks"),
    ],
    [
        State("model-type", "value"),
        State("optimizer-type", "value"),
        State("batch-size", "value"),
        State("learning-rate", "value"),
        State("max-iter", "value"),
        State("random-seed", "value"),
        State("update-rate", "value"),
    ],
)
def manage_image_generation(
    start_clicks,
    stop_clicks,
    reset_clicks,
    model_type,
    optimizer_type,
    batch_size,
    learning_rate,
    max_iter,
    random_seed,
    update_rate,
):
    global output_figure_queue, para_elbo_figure_queue, extra_layers, extra_layers_info, image_thread, stop_event, reset_flag, init_figure_flag

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    with lock:
        if triggered_id == "start-button":
            if init_figure_flag:
                init_figure_flag = False
                output_figure_queue.queue.clear()
                para_elbo_figure_queue.queue.clear()
            stop_event.clear()
        elif triggered_id == "stop-button":
            stop_event.set()
            return True  # Disable Interval component
        elif triggered_id == "reset-button" and (start_clicks != 0):
            stop_event.set()
            init_figure_flag = False
            reset_flag = True
            image_thread.join()
            output_figure_queue.queue.clear()
            para_elbo_figure_queue.queue.clear()
            stop_event.clear()
            init_figure_flag = True

        reset_flag = False
        if image_thread is not None and image_thread.is_alive() and not init_finished:
            init_figure_flag = True
        elif (init_figure_flag and image_thread is None) or (
            init_figure_flag and not image_thread.is_alive()
        ):
            image_thread = threading.Thread(
                target=image_generator,
                args=(
                    model_type,
                    optimizer_type,
                    batch_size,
                    learning_rate,
                    max_iter,
                    random_seed,
                    update_rate,
                ),
            )
            image_thread.start()

    return False


@app.callback(
    [Output("output-graph", "figure"), Output("para_elbo-graph", "figure")],
    [Input("interval-component", "n_intervals")],
)
def update_graph(n_intervals):
    global last_output_figure, last_para_elbo_figure, output_figure_queue, para_elbo_figure_queue
    if not output_figure_queue.empty():
        last_output_figure = output_figure_queue.get()
    if not para_elbo_figure_queue.empty():
        last_para_elbo_figure = para_elbo_figure_queue.get()
    return (
        last_output_figure,
        last_para_elbo_figure,
    )  # Return the last figure, updated or not


@app.callback(
    Output("layers-info", "value"),
    [
        Input("add-button", "n_clicks"),
        Input("remove-button", "n_clicks"),
    ],
    [
        State("layer-type", "value"),
        State("activation_function-type", "value"),
        State("n_th-layer", "value"),
    ],
)
def update_extra_layers(
    add_clicks, remove_clicks, layer_type, activation_function_type, n_th_layer
):
    global extra_layers, extra_layers_info
    num_dim = 2
    warning_text = "---Try another combination!---"

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    def update_layer_info():
        """Update the numbering in the extra_layers_info list."""
        return [
            f"{i + 1}. {info.split('. ', 1)[1]}"
            for i, info in enumerate(extra_layers_info)
        ]

    if button_id == "add-button":
        new_layer = None
        new_layer_info = None
        # Layer options: PlanarLayer Tanh, PlanarLayer LeakyRelu, FullRankNormalLayer, MeanFieldNormalLayer
        if layer_type == "PlanarLayer" and activation_function_type == "Tanh":
            new_layer = PlanarLayer(num_dim, Tanh())
            new_layer_info = "PlanarLayer Tanh"
        elif layer_type == "PlanarLayer" and activation_function_type == "LeakyRelu":
            new_layer = PlanarLayer(num_dim, LeakyRelu())
            new_layer_info = "PlanarLayer LeakyRelu"
        elif layer_type == "FullRankNormalLayer" and activation_function_type == "None":
            new_layer = FullRankNormalLayer(num_dim)
            new_layer_info = "FullRankNormalLayer"
        elif (
            layer_type == "MeanFieldNormalLayer" and activation_function_type == "None"
        ):
            new_layer = MeanFieldNormalLayer(num_dim)
            new_layer_info = "MeanFieldNormalLayer"

        if new_layer and new_layer_info:
            if warning_text in extra_layers_info:
                extra_layers_info.remove(warning_text)
            index = n_th_layer - 1 if n_th_layer is not None else len(extra_layers)
            index = max(
                0, min(index, len(extra_layers))
            )  # Ensure index is within bounds
            extra_layers.insert(index, new_layer)
            extra_layers_info.insert(index, f"{index + 1}. {new_layer_info}")
            extra_layers_info = update_layer_info()  # Update the numbering
        elif warning_text not in extra_layers_info:
            extra_layers_info.append(warning_text)

    elif button_id == "remove-button":
        if warning_text in extra_layers_info:
            extra_layers_info.remove(warning_text)
        if extra_layers:
            index = n_th_layer - 1 if n_th_layer is not None else len(extra_layers)
            index = max(
                0, min(index, len(extra_layers) - 1)
            )  # Ensure index is within bounds
            extra_layers.pop(index)
            extra_layers_info.pop(index)
            extra_layers_info = update_layer_info()  # Update the numbering

    return "\n".join(map(str, extra_layers_info))


@app.callback(
    Output("init_status-label", "children"),
    [Input("interval-component", "n_intervals")],
)
def update_status_label(n_intervals):
    global init_finished
    return "You can start now!" if init_finished else "Initializing, please wait!"


if __name__ == "__main__":
    app.run_server(debug=True, port=8050, host="localhost")
