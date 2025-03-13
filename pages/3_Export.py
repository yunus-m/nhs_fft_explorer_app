import streamlit as st
import pandas as pd

st.title('Export')

st.write('Export data or plots')
st.write('Implementing CSV export for now')

st.download_button(
    label='Download CSV',
    data='1, 2, 3', #df.to_csv(),
    file_name='sl_test.csv',
    mime='text/csv'
)