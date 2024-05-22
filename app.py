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
queue_size = 400  # Maximum size of the image queue
output_figure_queue = queue.Queue(maxsize=queue_size)  # Queue to store generated images
para_elbo_figure_queue = queue.Queue(maxsize=queue_size)  # Queue to store generated images
last_output_figure = go.Figure()  # Global variable to store the last figure
last_para_elbo_figure = go.Figure()  # Global variable to store the last figure

image_thread = None
stop_event = threading.Event()  # Event to stop the image generation thread
reset_flag = False

app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#ffffff'}, children=[
    html.H1("Variational Inference Visualization", style={'textAlign': 'center'}),
    
    html.Div([
        dcc.Interval(
            id='interval-component',
            interval=250,
            n_intervals=0,
            disabled=True
        ),
        
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between'}, children=[
            html.Div(style={'flex': '0 0 15%'}, children=[
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
                        dcc.Input(id='batch-size', type='number', value=8, step=1),
                    ])
                ], style={'margin-bottom': '10px'}),
                
                html.Div([
                    html.Label('Learning Rate:', style={'margin-right': '10px'}),
                    html.Div([
                        dcc.Input(id='learning-rate', type='number', value=1e-3, step=0.001),
                    ])
                ], style={'margin-bottom': '10px'}),
                
                html.Div([
                    html.Label('Max Iterations:', style={'margin-right': '10px'}),
                    html.Div([
                        dcc.Input(id='max-iter', type='number', value=2000, step=1000),
                    ])
                ], style={'margin-bottom': '10px'}),
                
                html.Div([
                    html.Label('Random Seed:', style={'margin-right': '10px'}),
                    html.Div([
                        dcc.Input(id='random-seed', type='number', value=2, step=1),
                    ])
                ], style={'margin-bottom': '10px'}),
                
                html.Div([
                    html.Label('Update Rate:', style={'margin-right': '10px'}),
                    html.Div([
                        dcc.Input(id='update-rate', type='number', value=10, step=5),
                    ])
                ], style={'margin-bottom': '10px'}),
                
                html.Div([
                    html.Button('Start', id='start-button', n_clicks=0, style={'margin-right': '10px'}),
                    html.Button('Stop', id='stop-button', n_clicks=0, style={'margin-right': '10px'}),
                    html.Button('Reset', id='reset-button', n_clicks=0)
                ], style={'margin-bottom': '20px'}),
            ]),
            
            html.Div(style={'flex': '0 0 85%'}, children=[
                dcc.Graph(id='para_elbo-graph')
            ])
        ]),
        
        dcc.Graph(id='output-graph')
    ], style={'margin': '0 auto', 'width': '80%', 'max-width': '1200px'})
])


def image_generator(model_type, optimizer_type, batch_size, learning_rate, max_iter, random_seed, update_rate):
    global output_figure_queue, stop_event, reset_flag
    
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
    elbo_list = []
    variational_parameters_list = []
    while iteration_count <= max_iter:
        if stop_event.is_set() and reset_flag:
            break
        elif stop_event.is_set() and not reset_flag:
            pass
        else:
            elbo, elbo_gradient = elbo_model.evaluate_and_gradient(variational_parameters)
            variational_parameters = optimizer.step(variational_parameters, elbo_gradient)
            
            elbo_list.append(elbo)
            variational_parameters_list.append(variational_parameters[0].item())
            
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
                
                fig_output = make_subplots(rows=1, cols=2, subplot_titles=("Normalizing Flow PDF", "True PDF"))
                
                fig_output.add_trace(go.Histogram2d(
                    x=samples[:, 0].flatten(),
                    y=samples[:, 1].flatten(),
                    autobinx=False,
                    autobiny=False,
                    xbins=dict(start=-2, end=2, size=0.04),
                    ybins=dict(start=-2, end=3, size=0.05),
                    colorscale='Viridis',
                    colorbar=dict(title='Density', x=0.4)
                ), row=1, col=1)
                
                fig_output.add_trace(go.Contour(
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
                

                fig_output.update_layout(
                    title="Variational Inference Visualization --- Iteration: " + str(iteration_count),
                    title_x=0.5,
                    title_y=0.9,  # 根据需要调整此值以控制标题在垂直方向上的位置        
                    height=600,
                    width=1200,
                    margin=dict(l=0, r=0, t=100, b=100),
                    autosize=False,
                    xaxis_title="X Axis",
                    yaxis_title="Y Axis",
                    xaxis2_title="X Axis",
                    yaxis2_title="Y Axis",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    xaxis=dict(domain=[0, 0.4]),  
                    yaxis=dict(domain=[0, 1]),    
                    xaxis2=dict(domain=[0.6, 1]), 
                    yaxis2=dict(domain=[0, 1])    
                )
                
                fig_para_elbo = make_subplots(rows=1, cols=2, subplot_titles=("Variational Parameters --- Mean: {:.5f}".format(np.mean(variational_parameters)), "Elbo --- Absolute Value: {:.2f}".format(np.abs(elbo))))
                
                # fig_para_elbo.add_trace(go.Scatter(
                #     x=np.arange(len(variational_parameters_list)),
                #     y=np.array(variational_parameters_list),
                #     mode='lines',
                #     line=dict(color='red', width=2)
                # ), row=1, col=1)
                
                fig_para_elbo.add_trace(go.Scatter(
                    x=np.arange(len(variational_parameters)),
                    y=np.array(variational_parameters),
                    mode='markers',
                    line=dict(color='orange', width=2)
                ), row=1, col=1)
                
                fig_para_elbo.add_trace(go.Scatter(
                    x=np.arange(len(elbo_list)),
                    y=np.array(elbo_list),
                    mode='lines',
                    line=dict(color='blue', width=2)
                ), row=1, col=2)
                
                fig_para_elbo.update_layout(
                    title="Variational Parameters and Elbo --- Iteration: " + str(iteration_count),
                    title_x=0.5,
                    title_y=1,         
                    height=500,
                    width=900,
                    margin=dict(l=100, r=0, t=50, b=50),
                    autosize=False,
                    xaxis_title="Different Parameters",
                    yaxis_title="Value of Parameters",
                    xaxis2_title="Iteration",
                    yaxis2_title="Elbo Value",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    xaxis_range=[0, len(variational_parameters)],
                    yaxis_range=[-2, 2],
                    xaxis2_range=[0, max_iter],
                    yaxis2_range=[np.min(elbo_list), np.max(elbo_list)]
                )
                
                fig_para_elbo.update_xaxes(
                    showgrid=False,
                    gridwidth=1,
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='LightGray'
                )
                fig_para_elbo.update_yaxes(
                    showgrid=False,
                    gridwidth=1,
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='LightGray'
                )
                
                if output_figure_queue.qsize() < queue_size - 1:
                    output_figure_queue.put(fig_output)
                    para_elbo_figure_queue.put(fig_para_elbo)
                else:
                    output_figure_queue.put(fig_output)
                    para_elbo_figure_queue.put(fig_para_elbo)
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
    global stop_event, output_figure_queue, image_thread, reset_flag

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
        reset_flag = True
        stop_event.set()
        output_figure_queue.queue.clear()
        para_elbo_figure_queue.queue.clear()
        image_thread.join()
        stop_event.clear()
        image_thread = threading.Thread(target=image_generator, args=(model_type, optimizer_type, batch_size, learning_rate, max_iter, random_seed, update_rate))
        image_thread.start()
        return False  # Disable Interval component initially
    
    return True


@app.callback(
    [Output('output-graph', 'figure'),
     Output('para_elbo-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n_intervals):
    global last_output_figure, last_para_elbo_figure, output_figure_queue, para_elbo_figure_queue
    if not output_figure_queue.empty():
        last_output_figure = output_figure_queue.get()
    if not para_elbo_figure_queue.empty():
        last_para_elbo_figure = para_elbo_figure_queue.get()
    return last_output_figure, last_para_elbo_figure  # Return the last figure, updated or not

if __name__ == '__main__':
    app.run_server(debug=True, port=8050, host='localhost')
