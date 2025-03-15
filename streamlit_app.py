import streamlit as st

pg = st.navigation(['1_Load.py', '2_Explore.py', '3_Export.py', '4_About.py'])
pg.run()

# st.title('Landing page')
# st.write('Use this app to process, explore, and export FFT data.')