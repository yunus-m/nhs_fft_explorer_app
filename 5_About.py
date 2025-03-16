import streamlit as st

st.title('About')

st.markdown(
    """
    This is a simple streamlit app for exploring and exporting sentiment in FFT feedback.
             
    Use the **Load** page to load and process data.
    
    Use the **Explore themes** page to explore themes via word clouds.

    Use the **Explore model** page to explore model's confidence and entropy measures.

    Use the **Export** page to view annotated table of data and export the results as a spreadsheet.
    """
)