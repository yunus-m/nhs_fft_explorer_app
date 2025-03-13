
import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.title('Explore')  #st.header
st.write('Explore the scatter plots with different column colourings')

#Upload
df = None

uploaded_file = st.file_uploader('Load a CSV', type='csv')

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning('No data')

if df:
    fig = make_subplots(
        rows=2, cols=2,
        #shared_xaxes=True, shared_yaxes=True, #latter not quite as needed
        
        # specs=[
        #     [{'colspan': 2, 'rowspan': 3}, None, {}],
        #     [None, None, {}],
        #     [None, None, {}]
        # ],
        
        # column_widths=[0.45, 0.45, 0.1],
        # row_heights=[1, 1, 1],
        # horizontal_spacing=0.05, vertical_spacing=0.05
    )

    fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='markers', marker=dict(color=df['column1'])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='markers', marker=dict(color=df['column2'])), row=1, col=2)
    fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='markers', marker=dict(color=df['column3'])), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='markers', marker=dict(color=df['column4'])), row=2, col=2)

    #manually link all to same x/y
    for row, col in [[1, 1], [1, 2], [2, 1], [2, 2]]:
        fig.update_xaxes(matches='x', row=row, col=col)
        fig.update_yaxes(matches='y', row=row, col=col)

    #Remove grid and axes
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)

    fig.update_layout(
        title='',
        showlegend=False,
        height=600, width=1200
    )

    st.plotly_chart(fig)
