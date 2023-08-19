import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.io as pio

# define the feasible region boundary
x = np.linspace(0, 7, 500)
y1 = 6 - x  # x + y <= 6
y2 = 11 - 2*x  # 2x + y <= 8
y3 = 4.5 - 0.5*x

# find the overlapping region
overlap = np.minimum(y1, np.minimum(y2, y3))

# create a DataFrame for the boundary lines
df_boundary = pd.DataFrame({'x': x, 'y1': y1, 'y2': y2, 'y3': y3})

# create a DataFrame for the overlapping region
df_overlap = pd.DataFrame({'x': x, 'y': overlap})

# create the plot
fig = go.Figure()

# add the overlapping region
fig.add_trace(go.Scatter(
    x=df_overlap['x'],
    y=df_overlap['y'],
    fill='tozeroy',
    fillcolor='rgba(128,128,128,0.3)',
    line=dict(color='rgba(0,0,0,0)'),
    name='feasible region'
))

# add the boundary lines
fig.add_trace(go.Scatter(
    x=df_boundary['x'],
    y=df_boundary['y1'],
    mode='lines',
    line=dict(color='red'),
    name='𝑥₁ + 𝑥₂ ≤ 6'
))

fig.add_trace(go.Scatter(
    x=df_boundary['x'],
    y=df_boundary['y2'],
    mode='lines',
    line=dict(color='green'),
    name='2𝑥₁ + 𝑥₂ ≤ 11'
))

fig.add_trace(go.Scatter(
    x=df_boundary['x'],
    y=df_boundary['y3'],
    mode='lines',
    line=dict(color='blue'),
    name='𝑥₁ + 2𝑥₂ ≤ 9'
))

# set the axis limits
fig.update_layout(xaxis_title="𝑥₁", yaxis_title="𝑥₂", xaxis_range=[0, 6], yaxis_range=[0, 6], title={'text': 'Mozart Problem', 'x': 0.5, 'y': 0.9, 'xanchor': 'center', 'yanchor': 'top'})

# show the plot
fig.show()

# Save the plot as an SVG file
# Configure plotly.io to suppress MathJax pop-up message
config = {'displayModeBar': False, 'showLink': False}

# Save the plot as a high-resolution PDF
pio.write_image(fig, 'mozart_problem.pdf', engine='kaleido', width=800, height=600, config=config)

# Show the plot
fig.show(config=config)
