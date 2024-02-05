import streamlit as st
import pandas as pd
import pydeck as pdk

# Sample data points
data = pd.DataFrame({
    'latitude': [45.521, 47.621, 46.874],
    'longitude': [-122.671, -122.121, -123.874],
})

# Create a PyDeck layer with more visible settings
layer = pdk.Layer(
    'ScatterplotLayer',     # Type of layer
    data,
    get_position='[longitude, latitude]',
    get_color='[255, 0, 0, 160]',  # Bright red color, with some transparency
    get_radius=20000,  # Larger radius for visibility, in meters
)

# Define the initial view state for focusing the map
view_state = pdk.ViewState(
    latitude=45.52,
    longitude=-122.67,
    zoom=5,  # Adjust the zoom level to make sure all points are within view
    pitch=0,
)

# Render the map with the layer
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
