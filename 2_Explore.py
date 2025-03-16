
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from umap import UMAP
import textwrap

from spreadsheet_data_handling import sentiment_dict
from frontend_utils import discrete_plotly_colorscale


st.title('Explore')

col1, col2, col3 = st.columns(3)
with col1:
    neigh_slider = st.slider(
        'local vs global structure', min_value=3, max_value=50, value=15, step=2,
        help='Smaller values emphasise intracluster detail, whereas larger values focus on global structure'
    )

with col2:
    size_slider = st.slider('marker size', min_value=1, max_value=10, value=3, step=1)

with col3:
    opacity_slider = st.slider('opacity', min_value=0.1, max_value=1., value=1., step=0.1)



if hasattr(st.session_state, 'data_dict'):
    df = st.session_state.data_dict['df_tweaked']
    predictions = st.session_state.data_dict['predictions']
    class_probs = st.session_state.data_dict['class_probs']
    entropies = st.session_state.data_dict['entropies']
    descriptions = [sentiment_dict[p] for p in predictions]

    #UMAP
    proj = UMAP(neigh_slider, random_state=0).fit_transform(st.session_state.data_dict['embeddings'])
    proj_x, proj_y = proj.T

    #Update state for Export page
    st.session_state.data_dict.update({'descriptions': descriptions, 'proj': proj})
    

    #
    # Figures
    #
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0,)

    hovertext = [
        f'{desc}    #{ix}' + '<br>' + textwrap.fill(text=ans, width=50).replace('\n', '<br>')
        for desc, ans, ix in zip(descriptions, df.answer_clean, df.index)
    ]

    hoverlabel = dict(
        font=dict(size=12, family='Arial', weight='bold'),
        bgcolor='rgba(0, 0, 0, 1)',
        bordercolor='rgba(0, 0, 0, 1)'
    )
    
    scatter_props = dict(
        x=proj_x, y=proj_y,
        hovertext=hovertext, hoverlabel=hoverlabel, hoverinfo='text',
        mode='markers',
        opacity=opacity_slider
    )

    scatter1 = go.Scatter(
        **scatter_props,
        marker=dict(
            color=predictions,
            colorscale=discrete_plotly_colorscale('PiYG_r', 5),
            size=size_slider,
            colorbar={
                'title': 'sentiment',
                'lenmode': 'fraction',
                'thickness': 30,
                'len': 0.3,
                'y': 0.9,
                'tickmode': 'array',
                'tickvals': list(sentiment_dict.keys()),
                'ticktext': list(sentiment_dict.values()),
            },
            #/colorbar
        ),
        #/marker
    )

    scatter2 = go.Scatter(
        **scatter_props,
        marker=dict(
            color=class_probs,
            colorscale='hot',
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
            colorscale='PuBu_r',
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
else:
    st.warning('No data loaded')