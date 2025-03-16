import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from frontend_utils import style_sentiment_description, to_excel


st.title('Export')

if not hasattr(st.session_state, 'data_dict'):
    st.warning('No data loaded')
elif 'proj' not in st.session_state.data_dict:
    st.warning('Use the Explore tab first')
else:
    df = st.session_state.data_dict['df_tweaked']
    predictions = st.session_state.data_dict['predictions']
    class_probs = st.session_state.data_dict['class_probs']
    entropies = st.session_state.data_dict['entropies']
    
    umap_proj = st.session_state.data_dict['umap_proj']
    descriptions = st.session_state.data_dict['descriptions']
    
    final_df = (
        df.assign(
            prediction=predictions,
            sentiment=descriptions,
            probability=class_probs,
            entropy=entropies,
            **{'2D viz': ['({:.1f}, {:.1f})'.format(*xy) for xy in umap_proj]}
        )
        .sort_values('prediction', ascending=False)
        [['question_type', 'answer_clean', 'sentiment', 'probability', 'entropy', '2D viz']]
    )
    final_df_styled = (
        final_df
        .style
        .map(style_sentiment_description, subset='sentiment')
        .background_gradient(subset='probability', cmap='hot')
        .background_gradient(subset='entropy', cmap='PuBu_r')
        .format(subset=['probability', 'entropy'], precision=2)
    )
    st.dataframe(final_df_styled, use_container_width=True)

    #
    # Download buttons
    #
    st.download_button(
        label='Export as Excel',
        data=to_excel(final_df_styled),
        file_name='export.xlsx',
        mime="application/vnd.ms-excel",
    )

    st.download_button(
        label='Export as CSV',
        data=final_df.to_csv(),
        file_name='export.csv',
        mime='text/csv',
    )