import streamlit as st
import pandas as pd
import numpy as np

from spreadsheet_data_handling import tweak_raw_df
from model_utils import forward_pass_from_checkpoint

#Path to checkpointed model and tokenizer
model_dir = 'data/hfcheckpoint-tinybert-sst5-fft'
tokenizer_dir = 'data/hfcheckpoint-tokenizer'


st.title('Load and process FFT data')

#
# Data has been loaded - show
#
file_uploader_btn = st.file_uploader('Load FFT data', type='csv')
if file_uploader_btn:
    df = pd.read_csv(file_uploader_btn, index_col=0)
    st.write(f'**Original data** | {len(df)} rows, {df.shape[1]} columns. Showing first 50 rows.')
    st.dataframe(df.iloc[:50])

    df_tweaked = df.pipe(tweak_raw_df)[['Comment ID', 'question_type', 'answer_clean']]
else:
    st.warning('No data loaded')

#
# Show tweaked data
#
if file_uploader_btn and st.checkbox('Show formatted data'):
    st.write('**Formatted data**')
    st.dataframe(df_tweaked.iloc[:50])

#
# Run button
#
def run_processing_callback():
    with st.spinner('Processing data...', show_time=True):
        model_results = forward_pass_from_checkpoint(
            df_tweaked, model_dir, tokenizer_dir
        )
        st.success('Processing complete')

        #Write results to session_state
        st.session_state.data_dict = {'df_tweaked': df_tweaked} | model_results

if file_uploader_btn:
    st.button('Process data', on_click=run_processing_callback)
