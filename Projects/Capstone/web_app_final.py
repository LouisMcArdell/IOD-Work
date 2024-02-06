import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import folium
from streamlit_folium import st_folium


# Set Streamlit page configuration to use wide mode
st.set_page_config(layout='wide', page_title='NZ Crash Data Clustering')

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

if 'run_clustering' not in st.session_state:
    st.session_state['run_clustering'] = False        
   

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
    
    # Create the map with an initial tile layer
    map = folium.Map(location=map_center, zoom_start=16, tiles=None)  # Setting tiles=None to start with no tiles
    
    # Define different tile layers including CartoDB Positron and Esri Satellite
    tiles = [
        ('Esri Satellite', 
         'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', 
         {'attr': 'Esri', 'overlay': False, 'name': 'Esri Satellite'}),
        ('CartoDB Positron', 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png', 
         {'attr': 'CartoDB', 'overlay': False, 'name': 'CartoDB Positron'}),
    ]

    # Add tile layers to the map
    for name, tile_url, options in tiles:
        folium.TileLayer(tile_url, attr=options['attr'], name=options['name'], overlay=options['overlay']).add_to(map)


    # Add markers with tooltips as unique identifiers
    for idx, row in clustered_df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=severity_colors[row['crashSeverity']],
            fill=True,
            fill_color=severity_colors[row['crashSeverity']],
            tooltip=str(idx),  # Use the DataFrame index or a unique identifier column as tooltip
        ).add_to(map)
    
    # Add LayerControl to allow users to switch between layers
    folium.LayerControl().add_to(map)

    return map

def prepare_cluster_summary(df):
    # Count occurrences of each cluster
    cluster_counts = df['cluster'].value_counts().rename_axis('Cluster').reset_index(name='Crash Count')
    
    # Find the most common 'region' and 'urban' value for each cluster
    most_common_region = df.groupby('cluster')['region'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan).rename_axis('Cluster').reset_index(name='Region')
    most_common_urban = df.groupby('cluster')['urban'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan).rename_axis('Cluster').reset_index(name='Urban/Open Road')
    
    # Merge the aggregated dataframes
    summary_df = pd.merge(cluster_counts, most_common_region, on='Cluster')
    summary_df = pd.merge(summary_df, most_common_urban, on='Cluster')

    # Sort the summary DataFrame by 'Cluster' to ensure it is ordered by cluster number
    summary_df.sort_values(by='Cluster', inplace=True)
    
    # Set 'Cluster' as the index of the DataFrame
    summary_df.set_index('Cluster', inplace=True)
    
    return summary_df


def display_transposed_summary(cluster_df, cluster_number):
    """
    Displays a transposed summary for the selected cluster.

    Parameters:
    - cluster_df: DataFrame containing data for the selected cluster.
    - cluster_number: The number of the selected cluster.
    """
    # Aggregate data for the selected cluster
    most_common_region = cluster_df['region'].mode()[0] if not cluster_df['region'].mode().empty else 'N/A'
    most_common_urban = cluster_df['urban'].mode()[0] if not cluster_df['urban'].mode().empty else 'N/A'
    most_common_speed_limit = cluster_df['speedLimit'].mode()[0] if not cluster_df['speedLimit'].mode().empty else 'N/A'
    crash_count = len(cluster_df)
    center_latitude = cluster_df['latitude'].mean()
    center_longitude = cluster_df['longitude'].mean()

    # Generate Google Maps URL with a marker at the center point
    maps_url = f"https://www.google.com/maps?q={center_latitude},{center_longitude}&hl=en&z=17"

    # Generate a direct Google Street View URL
    street_view_url = f"http://maps.google.com/maps?q=&layer=c&cbll={center_latitude},{center_longitude}"

    # Create a Markdown string for the transposed table
    summary_md = f"""
    | Field               | Value     |
    |---------------------|-----------|
    | Cluster Number      | {cluster_number} |
    | Crash Count         | {crash_count} |
    | Region  | {most_common_region} |
    | Urban/Open Road     | {most_common_urban} |
    | Speed Limit | {most_common_speed_limit} |
    | View on Google Maps       | [Google Maps]({maps_url}) |
    | View on Google Street View| [Street View]({street_view_url}) |   
    """
    
    # Display the Markdown table in Streamlit
    st.markdown(summary_md)

# Streamlit app interface
st.title('NZ Crash Data Clustering')
st.subheader('Source: NZTA Crash Analysis System (CAS)')

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
    st.session_state['run_clustering'] = True
    filtered_df = df[df['crashSeverity'].isin(selected_severities)]
    if not filtered_df.empty:
        cluster_labels = perform_clustering(filtered_df, epsilon_km, min_samples)
        filtered_df['cluster'] = cluster_labels
        st.session_state['clustered_df'] = filtered_df[filtered_df['cluster'] != -1]

main_col, right_sidebar = st.columns([3, 3])

with main_col:
    if st.session_state.get('run_clustering', False) and 'clustered_df' in st.session_state and st.session_state['clustered_df'] is not None and not st.session_state['clustered_df'].empty:
        unique_clusters = np.sort(st.session_state['clustered_df']['cluster'].unique())
        selected_cluster = st.sidebar.selectbox("Select a cluster to view on map:", unique_clusters, key='selected_cluster')
        
        if selected_cluster is not None:
            selected_cluster_df = st.session_state['clustered_df'][st.session_state['clustered_df']['cluster'] == selected_cluster]
            result_map = plot_on_map(selected_cluster_df)
            map_response = st_folium(result_map, width=725, height=500)

            if map_response:
                # Explicitly handle None values for last_clicked and last_object_clicked
                last_clicked = map_response.get('last_clicked') or {}
                last_object_clicked = map_response.get('last_object_clicked') or {}

                # Now, safely check for 'lat' and 'lng' without causing TypeError
                last_clicked_lat = round(last_clicked.get('lat', 0), 6)
                last_clicked_lng = round(last_clicked.get('lng', 0), 6)
                last_clicked_lat_lng = (last_clicked_lat, last_clicked_lng)

                last_object_clicked_lat = round(last_object_clicked.get('lat', 0), 6)
                last_object_clicked_lng = round(last_object_clicked.get('lng', 0), 6)
                last_object_clicked_lat_lng = (last_object_clicked_lat, last_object_clicked_lng)

                # Check if last clicked location is different from last object clicked location
                if last_clicked_lat_lng != last_object_clicked_lat_lng:
                    # Different location clicked; reset selected crash ID
                    st.session_state['selected_crash_id'] = None
                else:
                    # Same location or a crash marker clicked; update selected crash ID if available
                    last_clicked_tooltip = map_response.get('last_object_clicked_tooltip')
                    st.session_state['selected_crash_id'] = last_clicked_tooltip if last_clicked_tooltip else st.session_state.get('selected_crash_id', None)
            
            # Prepare and display the summary for all clusters below the map
            st.write("Overall Cluster Summary")
            cluster_summary_df = prepare_cluster_summary(st.session_state['clustered_df'])
            st.dataframe(cluster_summary_df)  # Displaying the DataFrame for all clusters

with right_sidebar:

    # Display Cluster Summary
    if 'selected_cluster' in st.session_state and st.session_state['selected_cluster'] is not None:
        st.write(f"Details for Cluster {st.session_state['selected_cluster']}:")

        # Filter the DataFrame for the selected cluster
        selected_cluster_df = st.session_state['clustered_df'][st.session_state['clustered_df']['cluster'] == st.session_state['selected_cluster']]

        # Display cluster summary
        if not selected_cluster_df.empty:
            display_transposed_summary(selected_cluster_df, st.session_state['selected_cluster'])
        else:
            st.write("No details available for this cluster.")
    
    # Divider for visual separation
    st.markdown("---")
    

    # Conditional display based on whether a crash ID has been selected
    if 'selected_crash_id' in st.session_state and st.session_state['selected_crash_id']:
        selected_crash_id = st.session_state['selected_crash_id']
        
        # Ensure selected_cluster is available and valid
        if 'selected_cluster' in st.session_state and st.session_state['selected_cluster'] is not None:
            # Recreate selected_cluster_df based on the current selected cluster
            selected_cluster_df = st.session_state['clustered_df'][st.session_state['clustered_df']['cluster'] == st.session_state['selected_cluster']]

            # Attempt to find the selected crash within the selected cluster's DataFrame
            selected_crash_detail_df = selected_cluster_df[selected_cluster_df.index.astype(str) == selected_crash_id]

            if not selected_crash_detail_df.empty:
                st.write("Details for the selected crash:")
                st.dataframe(selected_crash_detail_df)
            else:
                # Display a message when details for the selected crash are not found
                st.write("No details found for the selected crash.")
    else:
        # Display a placeholder message when no crash is selected
        st.write("Click on a crash marker for details or elsewhere to clear selection.")

if 'clustered_df' in st.session_state and st.session_state['clustered_df'] is not None and st.session_state['clustered_df'].empty:
    st.warning("No clusters found. Adjust parameters and try again.")
elif not st.session_state.get('run_clustering', False):
    st.write("Select severities and parameters, then click 'Run Clustering' to display results.")