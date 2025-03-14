import streamlit as st
import pandas as pd

st.title('Export')

if not hasattr(st.session_state, 'data_dict'):
    st.warning('No data')
else:
    df = st.session_state.data_dict['df_tweaked']
    predictions = st.session_state.data_dict['predictions']
    class_probs = st.session_state.data_dict['class_probs']
    entropies = st.session_state.data_dict['entropies'] #/ np.log2(5) * 100
    
    proj = st.session_state.data_dict['proj']
    descriptions = st.session_state.data_dict['descriptions']
    
    desc_dtype = pd.CategoricalDtype(
        ['very positive', 'positive', 'neutral', 'negative', 'very negative'],
        ordered=True
    )

    st.dataframe(
        df
        .assign(
            sentiment=predictions,
            description=descriptions,#pd.Series(descriptions, dtype=desc_dtype),
            probability=class_probs,
            entropy=entropies
        )
        [['question_type', 'answer_clean', 'sentiment', 'description', 'probability', 'entropy']]
        .sort_values('sentiment', ascending=False)
        .style
        .background_gradient(subset='sentiment', cmap='PiYG_r')
        .background_gradient(subset='probability', cmap='Reds')
        .background_gradient(subset='entropy', cmap='Purples')
        .format(subset=['probability', 'entropy'], precision=2),
        
        use_container_width=True
    )

st.download_button(
    label='Download CSV',
    data='1, 2, 3', #df.to_csv(),
    file_name='sl_test.csv',
    mime='text/csv'
)