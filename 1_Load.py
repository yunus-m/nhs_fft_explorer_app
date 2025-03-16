import streamlit as st
import pandas as pd
import numpy as np

from spreadsheet_data_handling import tweak_raw_df
from model_utils import forward_pass_from_checkpoint

#Path to checkpointed model and tokenizer
model_dir = 'data/hfcheckpoint-tinybert-sst5-fft'
tokenizer_dir = 'data/hfcheckpoint-tokenizer'


st.title('Load and process FFT data')

uploaded_file = st.file_uploader('Load FFT data', type='csv')
run_proc_btn = None

if uploaded_file:
    #
    # Data has been loaded - show
    #
    df = pd.read_csv(uploaded_file, index_col=0)
    st.toast(f'Loaded ' + uploaded_file.name)

    st.write(f'**Original data** | {len(df)} rows, {df.shape[1]} columns. Showing first 100 rows.')
    st.dataframe(df.iloc[:100])

    #
    # Show tweaked data
    #
    if st.checkbox('Show formatted data'):
        df_tweaked = df.pipe(tweak_raw_df)[['Comment ID', 'question_type', 'answer_clean']]
        st.write('**Formatted data**')
        st.dataframe(df_tweaked.iloc[:100])

    run_proc_btn = st.button('Process data')
else:
    st.warning('No data')


if run_proc_btn:
    with st.spinner('Processing data...', show_time=True):
        model_results = forward_pass_from_checkpoint(
            df_tweaked, model_dir, tokenizer_dir
        )
    st.success('Done!')

    st.session_state.data_dict = {
        'df_tweaked': df_tweaked,
        'embeddings': model_results['embeddings'],
        'predictions': model_results['predictions'],
        'class_probs': model_results['class_probs'],
        'entropies': model_results['entropies'],
    }