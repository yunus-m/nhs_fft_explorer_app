
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from umap import UMAP
import textwrap
from wordcloud import WordCloud

from frontend_utils import discrete_plotly_colorscale
from spreadsheet_data_handling import sentiment_dict

#
# Callbacks
#
def umap_callback():
    """Set flag to indicate UMAP should be run"""
    st.session_state.data_dict['umap_proj_pg2'] = None


st.title('Explore themes')

col1, col2, col3, col4 = st.columns(4)
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

with col4:
    wordcloud_slider = st.slider('word cloud detail', min_value=5, max_value=50, value=15, step=10)

if hasattr(st.session_state, 'data_dict'):
    data_dict = st.session_state.data_dict
    df = data_dict['df_tweaked']
    predictions = data_dict['predictions']
    descriptions = data_dict['descriptions']
    embeddings = data_dict['embeddings']
    
    #Run UMAP and save to session_state
    if 'umap_proj_pg2' not in data_dict or data_dict['umap_proj_pg2'] is None:
        umap_proj = UMAP(neigh_slider, random_state=0, n_jobs=1).fit_transform(embeddings)
        st.session_state.data_dict['umap_proj_pg2'] = umap_proj
    umap_x, umap_y = st.session_state.data_dict['umap_proj_pg2'].T

    #
    # Create figure
    #
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
                'thickness': 25,
                'len': 0.8,
                'y': 0.5,
                'tickmode': 'array',
                'tickvals': list(sentiment_dict.keys()),
                'ticktext': list(sentiment_dict.values()),
            },
            #/colorbar
        ),
        #/marker
    )
    fig = go.Figure(scatter1)
    
    #Remove grid and axes
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)

    fig.update_layout(title='', showlegend=False, height=600)

    #
    # Render figure
    #
    event_data = st.plotly_chart(fig, on_select='rerun', use_container_width=True, selection_mode='lasso')

    # st.code(f"{len(event_data['selection']['point_indices'])} points selected")
    #
    # Word cloud on selection event
    #
    if event_data and len(event_data['selection']['point_indices']):
        ixs = event_data['selection']['point_indices']

        #Clean text and some additional stopword removal
        text = ' '.join(df.iloc[ixs].answer_clean.str.lower()).replace('.', ' ')
        text = text.replace('nothing', '').replace('none', '')
        
        #Build word cloud
        wc = WordCloud(
            min_font_size=4,
            max_font_size=200,
            max_words=wordcloud_slider,
            colormap='coolwarm',

            width=800, height=400,
            background_color='black',
            contour_color='black',
            mode='RGBA'
        ).generate(text)

        f, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='spline16')
        ax.axis('off')

        st.pyplot(f)
    else:
        st.warning('Select data to view a wordcloud')
else:
    st.warning('No data loaded')