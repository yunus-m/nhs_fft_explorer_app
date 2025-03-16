
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from umap import UMAP
import textwrap

from frontend_utils import discrete_plotly_colorscale
from spreadsheet_data_handling import sentiment_dict


#
# Callbacks
#
def umap_callback():
    """Set flag to indicate UMAP should be run"""
    st.session_state.data_dict['umap_proj_pg3'] = None


st.title('Explore model')

col1, col2, col3 = st.columns(3)
with col1:
    neigh_slider = st.slider(
        'local vs global structure',
        min_value=3, max_value=50, value=15, step=2,
        on_change=umap_callback,
        help='Smaller values emphasise intracluster detail, whereas larger values focus on global structure',
    )

with col2:
    size_slider = st.slider('marker size', min_value=1, max_value=10, value=3, step=1)

with col3:
    opacity_slider = st.slider('marker opacity', min_value=0.1, max_value=1., value=1., step=0.1)

if hasattr(st.session_state, 'data_dict'):
    data_dict = st.session_state.data_dict
    df = data_dict['df_tweaked']
    predictions = data_dict['predictions']
    class_probs = data_dict['class_probs']
    entropies = data_dict['entropies']
    descriptions = data_dict['descriptions']
    embeddings = data_dict['embeddings']

    #Run UMAP and save to session_state
    if 'umap_proj_pg3' not in data_dict or data_dict['umap_proj_pg3'] is None:
        umap_proj = UMAP(neigh_slider, random_state=0, n_jobs=1).fit_transform(embeddings)
        st.session_state.data_dict['umap_proj_pg3'] = umap_proj
    umap_x, umap_y = st.session_state.data_dict['umap_proj_pg3'].T
    
    
    #
    # Create figures
    #
    fig = make_subplots(cols=1, rows=3, vertical_spacing=0)

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
        x=umap_x, y=umap_y,
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
    fig.add_trace(scatter1, col=1, row=1)
    fig.add_trace(scatter2, col=1, row=2)
    fig.add_trace(scatter3, col=1, row=3)
    
    #manually link all to same x/y
    for col, row in [[1, 1], [1, 2], [1, 3]]:
        fig.update_xaxes(matches='x', col=col, row=row)
        fig.update_yaxes(matches='y', col=col, row=row)

    #Remove grid and axes
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)

    fig.update_layout(title='', showlegend=False, height=1000)

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning('No data loaded')