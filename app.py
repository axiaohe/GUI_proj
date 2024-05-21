import numpy as np
from models import GaussianModel, RezendeModel, RosenbrockModel, ELBOModel
from stochastic_optimizers import Adam, Adamax, RMSprop, SGD
from variational_distributions.normalizing_flow import (
    NormalizingFlowVariational, PlanarLayer, Tanh, LeakyRelu, FullRankNormalLayer,
    MeanFieldNormalLayer
)
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

# Define global variables
global_variational_parameters = None
global_history = None
global_iteration_count = 0

app.layout = html.Div([
    html.H1("Variational Inference Visualization"),
    html.Div([
        dcc.Interval(
            id='interval-component',
            interval=0.25*1000,  # 1 second interval
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
        html.Label('Update Rate:'),
        dcc.Input(id='update-rate', type='number', value=2),
        html.Button('Run', id='run-button', n_clicks=0),
        dcc.Graph(id='output-graph')
    ])
])

# Enable/Disable interval component
@app.callback(
    Output('interval-component', 'disabled'),
    [Input('run-button', 'n_clicks')]
)
def enable_interval(n_clicks):
    if n_clicks:
        # global global_variational_parameters, global_history
        # global_variational_parameters = None
        # global_history = None
        return False  # Enable Interval component
    return True  # Disable Interval component

# change the refresh rate of the graph
@app.callback(
    Output('interval-component', 'interval'),
    [Input('update-rate', 'value')]
)
def change_refresh_rate(update_rate):
    return update_rate * 1000

# Update graph callback
@app.callback(
    Output('output-graph', 'figure'),
    [Input('interval-component', 'n_intervals')],
    [
        Input('model-type', 'value'),
        Input('optimizer-type', 'value'),
        Input('batch-size', 'value'),
        Input('learning-rate', 'value'),
        Input('max-iter', 'value'),
        Input('random-seed', 'value'),
        Input('update-rate', 'value')
    ]
)
def update_graph(n_intervals, model_type, optimizer_type, batch_size, learning_rate, max_iter, random_seed, update_rate):
    fig = generate_figure(n_intervals, model_type, optimizer_type, batch_size, learning_rate, max_iter, random_seed, update_rate)
    return fig

def generate_figure(n_intervals, model_type, optimizer_type, batch_size, learning_rate, max_iter, random_seed, update_rate):
    global global_variational_parameters, global_history, global_iteration_count
    
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
    
    if global_variational_parameters is None:
        global_variational_parameters = variational_distribution.initialize_variational_parameters()
    if global_history is None:
        global_history = {'variational_parameters': [], 'elbo': []}

    print('Iteration', global_iteration_count)
    for i in range(max_iter):  # Adjust the step count as needed
        elbo, elbo_gradient = elbo_model.evaluate_and_gradient(global_variational_parameters)
        global_history['variational_parameters'].append(global_variational_parameters.copy())
        global_history['elbo'].append(elbo.copy())
        global_variational_parameters = optimizer.step(global_variational_parameters, elbo_gradient)
        
        global_iteration_count += 1
        if (i+1) % (update_rate * 20) == 0 or global_iteration_count >= max_iter:
            break
            
    xlin = np.linspace(-2, 2, 100)
    ylin = np.linspace(-2, 3, 100)
    X, Y = np.meshgrid(xlin, ylin)
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    samples = variational_distribution.draw(global_variational_parameters, n_draws=100_000)

    true_pdf = np.exp(model.evaluate(positions))
    levels = 10

    # Create the first 2D histogram trace
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
        colorbar=dict(title='PDF', x=0.95) 
    ), row=1, col=2)
    
    # Update layout to set axis labels, titles, and background color
    fig.update_layout(
        title="Variational Inference Visualization",
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        plot_bgcolor='rgba(0,0,0,0)',  # Set background color to transparent
        paper_bgcolor='rgba(0,0,0,0)',  # Set paper background color to transparent
        showlegend=False
    )

    # Update x and y axes to have the same styling
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
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
