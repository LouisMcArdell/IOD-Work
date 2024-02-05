import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import folium
from streamlit_folium import st_folium

# Custom CSS to enable scrolling in the sidebar (or any container)
st.markdown("""
<style>
.scrollable-container {
    height: 300px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

# Define a key for the map interaction to prevent reruns
MAP_KEY = 'map_interaction'
CENTER_START = [-41, 174]
ZOOM_START = 6

# Initialize the session state for map interaction
if MAP_KEY not in st.session_state:
    st.session_state[MAP_KEY] = False

# Initialize session state variables if they don't already exist
if 'clustered_df' not in st.session_state:
    st.session_state['clustered_df'] = None

if "center" not in st.session_state:
        st.session_state["center"] = CENTER_START
if "zoom" not in st.session_state:
        st.session_state["zoom"] = ZOOM_START
   

@st.cache_data
def load_data(url):
    return pd.read_parquet(url)

# URL pointing to the raw version of the Parquet file on GitHub
url = 'https://github.com/LouisMcArdell/IOD-Work/raw/main/Projects/Capstone/app_data.parquet'

# Read the Parquet file into a DataFrame using the cached load_data function
df = load_data(url)

def perform_clustering(subset_df, epsilon_km, min_samples):
    kms_per_radian = 6371.0088
    epsilon = epsilon_km / kms_per_radian
    coords = subset_df[['latitude', 'longitude']].values
    db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    return db.labels_

def plot_on_map(clustered_df):
    severity_colors = {
        'Non-Injury Crash': '#1f77b4',
        'Minor Crash': '#ffec8b',
        'Serious Crash': '#d62728', 
        'Fatal Crash': '#4d4d4d'
    }
    map_center = [clustered_df['latitude'].mean(), clustered_df['longitude'].mean()]
    map = folium.Map(location=map_center, zoom_start=14)  # Adjust zoom_start for closer view
    
    for idx, row in clustered_df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=severity_colors[row['crashSeverity']],
            fill=True,
            fill_color=severity_colors[row['crashSeverity']],
            popup=f"Cluster: {row['cluster']}"
        ).add_to(map)
    
    return map

def display_map(map_object):
    """
    Displays a Folium map in Streamlit using st_folium.

    Parameters:
    - map_object: A Folium map object to display.
    """
    # Display the Folium map in Streamlit
    st_folium(map_object, width=725, height=500)

# Streamlit app interface
st.title('Crash Data Clustering')

with st.sidebar:
    # Checkboxes for crash severity selection
    severities = ['Non-Injury Crash', 'Minor Crash', 'Serious Crash', 'Fatal Crash']
    selected_severities = [severity for severity in severities if st.checkbox(severity, key=severity)]

    # Inputs for epsilon and min_samples
    epsilon_km = st.number_input('Epsilon distance in km (e.g., 0.03 for 30m)', value=0.03, step=0.01)
    min_samples = st.number_input('Min samples', value=150, step=1)

    # Run button
    run_clustering = st.button('Run Clustering')

if run_clustering:
    filtered_df = df[df['crashSeverity'].isin(selected_severities)]
    if not filtered_df.empty:
        cluster_labels = perform_clustering(filtered_df, epsilon_km, min_samples)
        filtered_df['cluster'] = cluster_labels
        # Update session state with filtered DataFrame excluding noise (-1 label)
        st.session_state['clustered_df'] = filtered_df[filtered_df['cluster'] != -1]

# Main content area for displaying selected cluster map and summary
if not st.session_state['clustered_df'].empty:
    # Dropdown for selecting a cluster moved to sidebar to ensure selection can happen anytime
    selected_cluster = st.sidebar.selectbox("Select a cluster to view on map:", st.session_state['clustered_df']['cluster'].unique(), key='selected_cluster')

    selected_cluster_df = st.session_state['clustered_df'][st.session_state['clustered_df']['cluster'] == selected_cluster]
    result_map = plot_on_map(selected_cluster_df)
    display_map(result_map)

    # Display the cluster summary under the map
    st.write("Cluster Summary")
    st.table(st.session_state['clustered_df']['cluster'].value_counts().sort_index().rename_axis('Cluster').reset_index(name='Count'))
else:
    if run_clustering:
        st.warning("No clusters found. Adjust parameters and try again.")
    else:
        st.write("Select severities and parameters, then click 'Run Clustering' to display results.")