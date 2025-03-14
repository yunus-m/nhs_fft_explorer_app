
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.title('Explore')  #st.header
st.write('Explore the scatter plots with different column colourings')

col1, col2, col3 = st.columns(3)
with col1:
    neigh_slider = st.slider('local vs global relations', min_value=3, max_value=50, value=15, step=2)
with col2:
    size_slider = st.slider('marker size', min_value=1, max_value=10, value=3, step=1)
with col3:
    opacity_slider = st.slider('opacity', min_value=0.1, max_value=1., value=1., step=0.1)

if hasattr(st.session_state, 'data_dict'):
    from umap import UMAP

    df = st.session_state.data_dict['df_tweaked']

    proj = UMAP(neigh_slider, random_state=0).fit_transform(st.session_state.data_dict['embeddings'])
    proj_x, proj_y = proj.T
    
    predictions = st.session_state.data_dict['predictions']
    class_probs = st.session_state.data_dict['class_probs']
    entropies = st.session_state.data_dict['entropies'] #/ np.log2(5) * 100
    descriptions = [{0: 'very positive', 1: 'positive', 2: 'neutral', 3: 'negative', 4: 'very negative'}[p] for p in predictions]
    
    fig = make_subplots(
        rows=3, cols=1,
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

    hovertext = [f'{desc} <br> {ans}' for desc, ans in zip(descriptions, df.answer_clean)]
    scatter_props = dict(x=proj_x, y=proj_y, hovertext=hovertext, mode='markers', opacity=opacity_slider)

    scatter1 = go.Scatter(
        **scatter_props,
        marker=dict(
            color=predictions,
            colorscale='PiYG_r',
            size=size_slider,
            colorbar={
                'title': 'sentiment',
                'lenmode': 'fraction',
                'len': 0.3,
                'y': 0.9,
                'tickmode': 'array',
                'tickvals': [0, 1, 2, 3, 4],
                'ticktext': ['very positive', 'positive', 'neutral', 'negative', 'very negative']
            },
            #/colorbar
        ),
        #/marker
    )

    scatter2 = go.Scatter(
        **scatter_props,
        marker=dict(
            color=class_probs,
            colorscale='Reds',
            size=size_slider,
            colorbar={
                'title': 'probability',
                'lenmode': 'fraction',
                'len': 0.3,
                'y': 0.55,
            },
            #/colorbar
        ),
        #/marker
    )

    scatter3 = go.Scatter(
        **scatter_props,
        marker=dict(
            color=entropies,
            colorscale='Purples',
            size=size_slider,
            colorbar={
                'title': 'entropy',
                'lenmode': 'fraction',
                'len': 0.3,
                'y': 0.2,
            },
            #/colorbar
        ),
        #/marker
    )
    
    #Layout
    fig.add_trace(scatter1, row=1, col=1)
    fig.add_trace(scatter2, row=2, col=1)
    fig.add_trace(scatter3, row=3, col=1)
    
    #manually link all to same x/y
    for row, col in [[1, 1], [2, 1], [3, 1]]:
        fig.update_xaxes(matches='x', row=row, col=col)
        fig.update_yaxes(matches='y', row=row, col=col)

    #Remove grid and axes
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)

    fig.update_layout(
        title='',
        showlegend=False,
        height=1000, #width=100
    )

    st.plotly_chart(fig, use_container_width=True)

