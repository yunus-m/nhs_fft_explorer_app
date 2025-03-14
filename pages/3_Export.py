import streamlit as st
import pandas as pd

from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from spreadsheet_data_handling import sentiment_dict


#
# Helper functions
#
def style_sentiment_description(val):
    cmap = plt.get_cmap('PiYG_r', 5)
    mapping = {
        description: to_hex( cmap(i/(cmap.N-1)) ) for i, description in enumerate(sentiment_dict.values())
    }
    return f'background-color: {mapping[val]}; color: {"white" if val != "neutral" else "black"}'


def to_excel(df):
    buffer = BytesIO()

    with pd.ExcelWriter(buffer) as writer:
        df.to_excel(writer)
    buffer.seek(0)

    return buffer


#
# Page
#
st.title('Export')

if not hasattr(st.session_state, 'data_dict'):
    st.warning('No data loaded')
elif 'proj' not in st.session_state.data_dict:
    st.warning('Use the Explore tab first')
else:
    df = st.session_state.data_dict['df_tweaked']
    predictions = st.session_state.data_dict['predictions']
    class_probs = st.session_state.data_dict['class_probs']
    entropies = st.session_state.data_dict['entropies'] #/ np.log2(5) * 100
    
    proj = st.session_state.data_dict['proj']
    descriptions = st.session_state.data_dict['descriptions']
    
    st.dataframe(
        df
        .assign(
            prediction=predictions,
            sentiment=descriptions,
            probability=class_probs,
            entropy=entropies,
            **{'2D viz': ['({:.1f}, {:.1f})'.format(*xy) for xy in proj]}
        )
        .sort_values('prediction', ascending=False)
        [['question_type', 'answer_clean', 'sentiment', 'probability', 'entropy', '2D viz']]
        
        .style
        .map(style_sentiment_description, subset='sentiment')
        .background_gradient(subset='probability', cmap='hot')
        .background_gradient(subset='entropy', cmap='PuBu_r')
        .format(subset=['probability', 'entropy'], precision=2),
        
        use_container_width=True
    )

    #
    # Download buttons
    #
    st.download_button(
        label='Export as Excel',
        data=to_excel(df),
        file_name='export.xlsx',
        mime="application/vnd.ms-excel",
    )

    st.download_button(
        label='Export as CSV',
        data=df.to_csv(),
        file_name='export.csv',
        mime='text/csv',
    )