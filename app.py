import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import threading
import time
import queue

from models import GaussianModel, RezendeModel, RosenbrockModel, ELBOModel
from stochastic_optimizers import Adam, Adamax, RMSprop, SGD
from variational_distributions.normalizing_flow import (
    NormalizingFlowVariational, PlanarLayer, Tanh, LeakyRelu, FullRankNormalLayer,
    MeanFieldNormalLayer)

# Define global variables
image_queue = queue.Queue(maxsize=400)  # Queue to store generated images
stop_event = threading.Event()  # Event to stop the image generation thread
last_figure = go.Figure()  # Global variable to store the last figure

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Variational Inference Visualization"),
    html.Div([
        dcc.Interval(
            id='interval-component',
            interval=200,  # 100ms interval for animation effect
            n_intervals=0
        ),
        html.Label('Model Type:'),
        dcc.Dropdown(
            id='model-type',
            options=[
                {'label': 'Gaussian Model', 'value': 'GaussianModel'},
                {'label': 'RezendeModel', 'value': 'RezendeModel'},
                {'label': 'RosenbrockModel', 'value': 'RosenbrockModel'}
            ],
            value='GaussianModel'
        ),
        html.Label('Optimizer Type:'),
        dcc.Dropdown(
            id='optimizer-type',
            options=[
                {'label': 'Adam', 'value': 'Adam'},
                {'label': 'Adamax', 'value': 'Adamax'},
                {'label': 'RMSprop', 'value': 'RMSprop'},
                {'label': 'SGD', 'value': 'SGD'}
            ],
            value='Adam'
        ),
        html.Label('Batch Size:'),
        dcc.Input(id='batch-size', type='number', value=8),
        html.Label('Learning Rate:'),
        dcc.Input(id='learning-rate', type='number', value=1e-3),
        html.Label('Max Iterations:'),
        dcc.Input(id='max-iter', type='number', value=10000000),
        html.Label('Random Seed:'),
        dcc.Input(id='random-seed', type='number', value=2),
        html.Button('Run', id='run-button', n_clicks=0),
        dcc.Graph(id='output-graph')
    ])
])

def image_generator(model_type, optimizer_type, batch_size, learning_rate, max_iter, random_seed):
    global image_queue, stop_event
    
    np.random.seed(random_seed)
    num_dim = 2

    layers = [
        MeanFieldNormalLayer(num_dim),
        PlanarLayer(num_dim, Tanh()),
        FullRankNormalLayer(num_dim),
        PlanarLayer(num_dim, LeakyRelu()),
    ]
    layers += [PlanarLayer(num_dim, Tanh()) for _ in range(12)]

    variational_distribution = NormalizingFlowVariational(layers=layers)
    
    if model_type == 'GaussianModel':
        model = GaussianModel() 
    elif model_type == 'RezendeModel':
        model = RezendeModel()
    else:
        model = RosenbrockModel()
        
    elbo_model = ELBOModel(
        model=model,
        variational_distribution=variational_distribution,
        n_samples_per_iter=batch_size
    )
    
    if optimizer_type == 'Adam':
        optimizer = Adam(
            learning_rate=learning_rate,
            optimization_type='max'
        )
    elif optimizer_type == 'Adamax':
        optimizer = Adamax(
            learning_rate=learning_rate,
            optimization_type='max'
        )
    elif optimizer_type == 'RMSprop':
        optimizer = RMSprop(
            learning_rate=learning_rate,
            optimization_type='max'
        )
    else:
        optimizer = SGD(
            learning_rate=learning_rate,
            optimization_type='max'
        )
    
    variational_parameters = variational_distribution.initialize_variational_parameters()

    iteration_count = 0
    while not stop_event.is_set() and iteration_count < max_iter:
        elbo, elbo_gradient = elbo_model.evaluate_and_gradient(variational_parameters)
        variational_parameters = optimizer.step(variational_parameters, elbo_gradient)
        
        if iteration_count < 100:
            factor = (iteration_count / 10) + 1
        else:
            factor = 10
        
        if iteration_count % factor == 0:
            xlin = np.linspace(-2, 2, 100)
            ylin = np.linspace(-2, 3, 100)
            X, Y = np.meshgrid(xlin, ylin)
            positions = np.vstack([X.ravel(), Y.ravel()]).T
            samples = variational_distribution.draw(variational_parameters, n_draws=100_000)
            
            true_pdf = np.exp(model.evaluate(positions))
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Sample Histogram", "True PDF"))
            
            fig.add_trace(go.Histogram2d(
                x=samples[:, 0].flatten(),
                y=samples[:, 1].flatten(),
                autobinx=False,
                autobiny=False,
                xbins=dict(start=-2, end=2, size=0.04),
                ybins=dict(start=-2, end=3, size=0.05),
                colorscale='Viridis',
                colorbar=dict(title='Density', x=0.45)
            ), row=1, col=1)
            
            fig.add_trace(go.Contour(
                x=positions[:, 0],
                y=positions[:, 1],
                z=true_pdf,
                contours=dict(
                    start=0,
                    end=np.max(true_pdf),
                    size=0.1 * np.max(true_pdf),
                    coloring='heatmap'
                ),
                line=dict(
                    smoothing=0.85,
                    color='white'
                ),
                colorscale='Viridis',
                colorbar=dict(title='PDF', x=1.05)
            ), row=1, col=2)
            
            fig.update_layout(
                title="Variational Inference Visualization",
                height=600,
                width=1200,
                margin=dict(l=0, r=0, t=100, b=100),
                autosize=False,
                xaxis_title="X Axis",
                yaxis_title="Y Axis",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='LightGray'
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='LightGray'
            )

            if not image_queue.full():
                image_queue.put(fig)
            else:
                time.sleep(0.5)
        
        iteration_count += 1


@app.callback(
    Output('interval-component', 'disabled'),
    [Input('run-button', 'n_clicks')],
    [
        Input('model-type', 'value'),
        Input('optimizer-type', 'value'),
        Input('batch-size', 'value'),
        Input('learning-rate', 'value'),
        Input('max-iter', 'value'),
        Input('random-seed', 'value')
    ]
)
def start_stop_image_generation(n_clicks, model_type, optimizer_type, batch_size, learning_rate, max_iter, random_seed):
    global stop_event, image_queue
    
    if n_clicks:
        stop_event.clear()
        image_queue.queue.clear()
        threading.Thread(target=image_generator, args=(model_type, optimizer_type, batch_size, learning_rate, max_iter, random_seed)).start()
        return False  # Enable Interval component
    stop_event.set()
    return True  # Disable Interval component

@app.callback(
    Output('output-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n_intervals):
    print(image_queue.qsize())
    global last_figure
    if not image_queue.empty():
        last_figure = image_queue.get()
    return last_figure  # Return the last figure, updated or not

if __name__ == '__main__':
    app.run_server(debug=True, port=8050, host='localhost')