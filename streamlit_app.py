import streamlit as st

pg = st.navigation(['pages/1_Load.py', 'pages/2_Explore.py', 'pages/3_Export.py', 'pages/4_About.py'])
pg.run()

# st.title('Landing page')
# st.write('Use this app to process, explore, and export FFT data.')