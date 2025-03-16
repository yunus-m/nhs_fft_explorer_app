import streamlit as st

pg = st.navigation(['1_📂Load.py', '2_Explore_themes.py', '3_Explore_model.py', '4_Export.py', '5_About.py'])
pg.run()

# st.title('Landing page')
# st.write('Use this app to process, explore, and export FFT data.')