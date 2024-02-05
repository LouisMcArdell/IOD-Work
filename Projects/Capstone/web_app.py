import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import pydeck as pdk

# Initialize session state variables if they don't already exist
if 'clustered_df' not in st.session_state:
    st.session_state['clustered_df'] = None


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
        'Non-Injury Crash': [31, 119, 180, 140],  # Updated alpha value to 255 for clarity
        'Minor Crash': [255, 237, 139, 140],      
        'Serious Crash': [214, 39, 40, 140],      
        'Fatal Crash': [77, 77, 77, 140]          
    }

    # Apply color transformation directly in the DataFrame
    clustered_df['color'] = clustered_df['crashSeverity'].map(severity_colors)

    # Create a PyDeck layer for crashes
    layer = pdk.Layer(
        "ScatterplotLayer",
        clustered_df,
        get_position='[longitude, latitude]',
        get_color='color',  # Reference the color column directly
        get_radius=10,  # Adjusted for visibility
        pickable=True,
    )
    
    # Define the view state
    view_state = pdk.ViewState(
        latitude=clustered_df['latitude'].mean(),
        longitude=clustered_df['longitude'].mean(),
        zoom=6,
        pitch=0,
    )
    
    # Render the map
    return pdk.Deck(layers=[layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/light-v9')



# Streamlit app interface
st.title('Crash Data Clustering')

# Checkboxes for crash severity selection
severities = ['Non-Injury Crash', 'Minor Crash', 'Serious Crash', 'Fatal Crash']
selected_severities = [severity for severity in severities if st.checkbox(severity, key=severity)]

# Inputs for epsilon and min_samples
epsilon_km = st.number_input('Epsilon distance in km (e.g., 0.03 for 30m)', value=0.03, step=0.01)
min_samples = st.number_input('Min samples', value=150, step=1)

# Run button
if st.button('Run Clustering'):
    filtered_df = df[df['crashSeverity'].isin(selected_severities)]
    if not filtered_df.empty:
        cluster_labels = perform_clustering(filtered_df, epsilon_km, min_samples)
        clustered_df = filtered_df[cluster_labels != -1].copy()
        clustered_df['cluster'] = cluster_labels[cluster_labels != -1]
        st.session_state['clustered_df'] = clustered_df

# Display the results if available in session state
if st.session_state.get('clustered_df') is not None:
    clustered_df = st.session_state['clustered_df']
    cluster_summary = clustered_df['cluster'].value_counts().sort_index()
    st.write("Cluster summary (Cluster Label: Count):")
    st.write(cluster_summary)
    
    if len(cluster_summary) > 0:
        # Call your updated plot_on_map function
        deck = plot_on_map(clustered_df)
        
        # Display the map and capture the selected data
        selected_data = st.pydeck_chart(deck, use_container_width=True)
        
        # Display information about the selected point
        if selected_data:
            selected_id = selected_data['object']['id']
            selected_crash_info = clustered_df.loc[selected_id]
    
            # Extracting longitude and latitude values
            longitude = selected_crash_info['longitude']
            latitude = selected_crash_info['latitude']
    
            # Displaying selected crash information
            st.write("Selected Crash Information:")
            st.write(f"Longitude: {longitude}, Latitude: {latitude}")

    else:
        st.warning("No clusters found. Adjust parameters and try again.")
else:
    st.write("Select severities and parameters, then click 'Run Clustering' to display results.")