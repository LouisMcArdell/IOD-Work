import streamlit as st
import folium
from streamlit_folium import st_folium

def main():
    st.set_page_config("Simple Map Test", layout='wide')

    # Create a simple Folium map
    m = folium.Map(location=(33.748997, -84.387985), zoom_start=10, tiles="cartodbpositron")

    # Display the map in Streamlit
    st_folium(m, height=600)

if __name__ == "__main__":
    main()
