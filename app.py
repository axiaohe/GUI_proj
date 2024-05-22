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
    NormalizingFlowVariational, PlanarLayer, Tanh, LeakyRelu, FullRankNormalLayer,
    MeanFieldNormalLayer)

# Define global variables
image_queue = queue.Queue(maxsize=400)  # Queue to store generated images
stop_event = threading.Event()  # Event to stop the image generation thread
last_figure = go.Figure()  # Global variable to store the last figure
image_thread = None
reset_flag = False

app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#ffffff'}, children=[
    html.H1("Variational Inference Visualization", style={'textAlign': 'center'}),  # 居中标题
    
    html.Div([
        dcc.Interval(
            id='interval-component',
            interval=250,  # 250ms interval for animation effect
            n_intervals=0,
            disabled=True  # Initially disabled
        ),
                 
        html.Div([
            html.Label('Model Type:', style={'margin-right': '10px'}),
            dcc.Dropdown(
                id='model-type',
                options=[
                    {'label': 'Gaussian Model', 'value': 'GaussianModel'},
                    {'label': 'RezendeModel', 'value': 'RezendeModel'},
                    {'label': 'RosenbrockModel', 'value': 'RosenbrockModel'}
                ],
                value='GaussianModel'
            )
        ], style={'margin-bottom': '10px'}),
        
        html.Div([
            html.Label('Optimizer Type:', style={'margin-right': '10px'}),
            dcc.Dropdown(
                id='optimizer-type',
                options=[
                    {'label': 'Adam', 'value': 'Adam'},
                    {'label': 'Adamax', 'value': 'Adamax'},
                    {'label': 'RMSprop', 'value': 'RMSprop'},
                    {'label': 'SGD', 'value': 'SGD'}
                ],
                value='Adam'
            )
        ], style={'margin-bottom': '10px'}),
        
        html.Div([
            html.Label('Batch Size:', style={'margin-right': '10px'}),
            html.Div([
                dcc.Input(id='batch-size', type='number', value=8, step=1),  # 设置 step 属性
            ])
        ], style={'margin-bottom': '10px'}),
        
        html.Div([
            html.Label('Learning Rate:', style={'margin-right': '10px'}),
            html.Div([
                dcc.Input(id='learning-rate', type='number', value=1e-3, step=0.001),  # 设置 step 属性
            ])
        ], style={'margin-bottom': '10px'}),
        
        html.Div([
            html.Label('Max Iterations:', style={'margin-right': '10px'}),
            html.Div([
                dcc.Input(id='max-iter', type='number', value=10000, step=1000),  # 设置 step 属性
            ])
        ], style={'margin-bottom': '10px'}),
        
        html.Div([
            html.Label('Random Seed:', style={'margin-right': '10px'}),
            html.Div([
                dcc.Input(id='random-seed', type='number', value=2, step=1),  # 设置 step 属性
            ])
        ], style={'margin-bottom': '10px'}),
        
        html.Div([
            html.Label('Update Rate:', style={'margin-right': '10px'}),
            html.Div([
                dcc.Input(id='update-rate', type='number', value=10, step=5),  # 设置 step 属性
            ])
        ], style={'margin-bottom': '10px'}),
                
        html.Div([
            html.Button('Start', id='start-button', n_clicks=0, style={'margin-right': '10px'}),
            html.Button('Stop', id='stop-button', n_clicks=0, style={'margin-right': '10px'}),
            html.Button('Reset', id='reset-button', n_clicks=0)
        ], style={'margin-bottom': '20px'}),
        
        dcc.Graph(id='output-graph')
    ], style={'margin': '0 auto', 'width': '80%', 'max-width': '1200px'})  # 居中并限制宽度
])


def image_generator(model_type, optimizer_type, batch_size, learning_rate, max_iter, random_seed, update_rate):
    global image_queue, stop_event, reset_flag
    
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
    while iteration_count < max_iter:
        if stop_event.is_set() and reset_flag:
            break
        elif stop_event.is_set() and not reset_flag:
            pass
        else:
            print(iteration_count, image_queue.qsize())
            elbo, elbo_gradient = elbo_model.evaluate_and_gradient(variational_parameters)
            variational_parameters = optimizer.step(variational_parameters, elbo_gradient)
            
            # if iteration_count < update_rate * 10:
            #     factor = int(iteration_count / 10) + 1
            # else:
            #     factor = update_rate
            
            if iteration_count % update_rate == 0:
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
                    colorbar=dict(title='Density', x=0.448)
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
                    colorbar=dict(title='PDF', x=1.00)
                ), row=1, col=2)
                
                fig.update_layout(
                    #TODO: iteration count
                    title="Variational Inference Visualization --- Iteration: " + str(iteration_count),
                    title_x=0.5,
                    title_y=0.9,  # 根据需要调整此值以控制标题在垂直方向上的位置        
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
                #TODO: overflows
                if not image_queue.full():
                    image_queue.put(fig)
                else:
                    time.sleep(0.5)
            
            iteration_count += 1

@app.callback(
    Output('interval-component', 'disabled'),
    [
        Input('start-button', 'n_clicks'),
        Input('stop-button', 'n_clicks'),
        Input('reset-button', 'n_clicks')
    ],
    [
        State('model-type', 'value'),
        State('optimizer-type', 'value'),
        State('batch-size', 'value'),
        State('learning-rate', 'value'),
        State('max-iter', 'value'),
        State('random-seed', 'value'),
        State('update-rate', 'value')
    ]
)
def manage_image_generation(start_clicks, stop_clicks, reset_clicks, model_type, optimizer_type, batch_size, learning_rate, max_iter, random_seed, update_rate):
    global stop_event, image_queue, image_thread, reset_flag

    ctx = dash.callback_context

    if not ctx.triggered:
        return True

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'start-button':
        reset_flag = False
        stop_event.clear()
        if start_clicks == 1:
            image_thread = threading.Thread(target=image_generator, args=(model_type, optimizer_type, batch_size, learning_rate, max_iter, random_seed, update_rate))
            image_thread.start()
        return False  # Enable Interval component
    elif button_id == 'stop-button':
        reset_flag = False
        stop_event.set()
        return True  # Disable Interval component
    elif button_id == 'reset-button' and (start_clicks != 0):
        print('reset')
        reset_flag = True
        stop_event.set()
        image_queue.queue.clear()
        image_thread.join()
        stop_event.clear()
        image_thread = threading.Thread(target=image_generator, args=(model_type, optimizer_type, batch_size, learning_rate, max_iter, random_seed, update_rate))
        image_thread.start()
        return False  # Disable Interval component initially
    
    return True


@app.callback(
    Output('output-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n_intervals):
    global last_figure
    # print(image_queue.qsize())
    if not image_queue.empty():
        last_figure = image_queue.get()
    return last_figure  # Return the last figure, updated or not

if __name__ == '__main__':
    app.run_server(debug=True, port=8050, host='localhost')
