import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import folium
from streamlit_folium import st_folium


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
    map = folium.Map(location=map_center, zoom_start=6)
    
    # Create a FeatureGroup for the crashes
    crash_feature_group = folium.FeatureGroup(name='Crash Locations')
    
    for idx, row in clustered_df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=severity_colors[row['crashSeverity']],
            fill=True,
            fill_color=severity_colors[row['crashSeverity']],
            popup=f"Cluster: {row['cluster']}"
        ).add_to(crash_feature_group)
    
    # Optionally, if you want to add a LayerControl to toggle this group
    folium.LayerControl().add_to(map)
    
    return map, crash_feature_group


# Streamlit app interface
st.title('Crash Data Clustering')

# Checkboxes for crash severity selection
severities = ['Non-Injury Crash', 'Minor Crash', 'Serious Crash', 'Fatal Crash']
selected_severities = [severity for severity in severities if st.checkbox(severity, key=severity)]

# Inputs for epsilon and min_samples
epsilon_km = st.number_input('Epsilon distance in km (e.g., 0.03 for 30m)', value=0.03, step=0.01)
min_samples = st.number_input('Min samples', value=150, step=1)

# Initialize a flag in the session state if it's not already set
if 'button_pressed' not in st.session_state:
    st.session_state['button_pressed'] = False

# Run button
if st.button('Run Clustering'):
    # Filter the DataFrame based on selected severities
    filtered_df = df[df['crashSeverity'].isin(selected_severities)]
    
    if not filtered_df.empty:
        # Perform clustering and store the labels
        cluster_labels = perform_clustering(filtered_df, epsilon_km, min_samples)
        # Create a new DataFrame excluding noise (-1 label)
        clustered_df = filtered_df[cluster_labels != -1].copy()
        clustered_df['cluster'] = cluster_labels[cluster_labels != -1]
        # Update session state
        st.session_state['clustered_df'] = clustered_df

# Display the results if available in session state
if st.session_state['clustered_df'] is not None:
    clustered_df = st.session_state['clustered_df']
    
    # Display a summary of clusters
    cluster_summary = clustered_df['cluster'].value_counts().sort_index()
    st.write("Cluster summary (Cluster Label: Count):")
    st.write(cluster_summary)
    
    # Check if clustering was successful (i.e., more than one cluster)
    if len(cluster_summary) > 0:
        # Set the map interaction flag to True right before displaying the map
        st.session_state[MAP_KEY] = True

        # Display results on a folium map
        result_map, fg = plot_on_map(clustered_df)
        st_folium(
            result_map,
            center=st.session_state["center"],
            zoom=st.session_state["zoom"], 
            key="new",
            feature_group_to_add=fg, 
            width=725, 
            height=500
            )

        # Reset the map interaction flag to False right after displaying the map
        st.session_state[MAP_KEY] = False

        # If any interaction is detected, stop further execution of the app
        st.stop()
    else:
        st.warning("No clusters found. Adjust parameters and try again.")
else:
    st.write("Select severities and parameters, then click 'Run Clustering' to display results.")