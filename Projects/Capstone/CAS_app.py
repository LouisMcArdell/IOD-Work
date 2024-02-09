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
    map = folium.Map(location=map_center, zoom_start=17, tiles='CartoDB Positron', attr='CartoDB')
    
    # Define the Esri Satellite tile layer as an additional layer (not the default)
    folium.TileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite',
        overlay=False
    ).add_to(map)

    # Define the CartoDB Positron tile layer
    folium.TileLayer(
        'CartoDB Positron',
        attr='CartoDB',
        name='CartoDB Positron',
        overlay=False
    ).add_to(map)    

  
    # Add markers with tooltips as unique identifiers
    for idx, row in clustered_df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=severity_colors[row['crashSeverity']],
            fill=True,
            fill_color=severity_colors[row['crashSeverity']],
            fill_opacity=0.7,
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
    most_speed_limit = df.groupby('cluster')['speedLimit'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan).rename_axis('Cluster').reset_index(name='Speed Limit')
    # Merge the aggregated dataframes
    summary_df = pd.merge(cluster_counts, most_common_region, on='Cluster')
    summary_df = pd.merge(summary_df, most_common_urban, on='Cluster')
    summary_df = pd.merge(summary_df, most_speed_limit, on='Cluster')

    # Sort the summary DataFrame by 'Cluster' to ensure it is ordered by cluster number
    summary_df.sort_values(by='Cluster', inplace=True)
    
    # Set 'Cluster' as the index of the DataFrame
    summary_df.set_index('Cluster', inplace=True)
    
    return summary_df


def display_transposed_summary(cluster_df, cluster_number):
    # Basic cluster information
    most_common_region = cluster_df['region'].mode()[0] if not cluster_df['region'].mode().empty else 'N/A'
    most_common_urban = cluster_df['urban'].mode()[0] if not cluster_df['urban'].mode().empty else 'N/A'
    most_common_speed_limit = cluster_df['speedLimit'].mode()[0] if not cluster_df['speedLimit'].mode().empty else 'N/A'
    most_common_traffic_control = cluster_df['trafficControl'].mode()[0] if not cluster_df['trafficControl'].mode().empty else 'N/A'
    most_common_road_character = cluster_df['roadCharacter'].mode()[0] if not cluster_df['roadCharacter'].mode().empty else 'N/A'    
    crash_count = len(cluster_df)
    center_latitude = cluster_df['latitude'].mean()
    center_longitude = cluster_df['longitude'].mean()
    maps_url = f"https://www.google.com/maps?q={center_latitude},{center_longitude}&hl=en&z=17"
    street_view_url = f"http://maps.google.com/maps?q=&layer=c&cbll={center_latitude},{center_longitude}"

    # Weather conditions and percentages
    weather_counts = cluster_df['weatherA'].value_counts(normalize=True) * 100
    weather_summary = '\n'.join([f"| {condition} | {percentage:.2f}% |" for condition, percentage in weather_counts.items()])

    # Light conditions and percentages
    light_counts = cluster_df['light'].value_counts(normalize=True) * 100
    light_summary = '\n'.join([f"| {condition} | {percentage:.2f}% |" for condition, percentage in light_counts.items()])

    # Markdown table
    summary_md = f"""
| Field               | Value     |
|---------------------|-----------|
| Crash Count         | {crash_count} |
| Region              | {most_common_region} |
| Urban/Open Road     | {most_common_urban} |
| Speed Limit         | {most_common_speed_limit} |
| Traffic Control         | {most_common_traffic_control} |
| Road Character         | {most_common_road_character} |
**Weather Conditions** | **Percentages** |
{weather_summary}
**Light Conditions**   | **Percentages** |
{light_summary}
**Google Links**   |  |
| View on Google Maps | [Google Maps]({maps_url}) |
| View on Google Street View | [Street View]({street_view_url}) |
    """
    
    st.markdown(summary_md, unsafe_allow_html=True)





# Streamlit app interface
st.title('NZ Crash Data Clustering')
st.markdown('Source: [NZTA Crash Analysis System (CAS)](https://opendata-nzta.opendata.arcgis.com/documents/ae974fef37154108b9b1048471335e67/about)', unsafe_allow_html=True)


with st.sidebar:
    # Message for crash severity selection
    st.markdown("## Controls")
    st.markdown('<span style="font-size: 14px;">Select crash severities:</span>', unsafe_allow_html=True)
    # Checkboxes for crash severity selection
    severities = ['Non-Injury Crash', 'Minor Crash', 'Serious Crash', 'Fatal Crash']
    selected_severities = [severity for severity in severities if st.checkbox(severity, key=severity)]

    # Define regions based on value counts provided
    regions = [
        "Auckland Region",
        "Waikato Region",
        "Canterbury Region",
        "Wellington Region",
        "Bay of Plenty Region",
        "Manawatū-Whanganui Region",
        "Otago Region",
        "Northland Region",
        "Hawke's Bay Region",
        "Southland Region",
        "Taranaki Region",
        "Gisborne Region",
        "Marlborough Region",
        "Nelson Region",
        "Tasman Region",
        "West Coast Region",
        "Unknown"
    ]
    
    # Add option for selecting all regions
    regions_with_all_option = ["All"] + regions

    # Radio buttons for selecting a region
    selected_region = st.radio("Select a region:", options=regions_with_all_option, key='region')

    # Inputs for epsilon and min_samples
    epsilon_km = st.number_input('Epsilon distance in km (e.g., 0.03 for 30m)', value=0.03, step=0.01)
    min_samples = st.number_input('Min samples', value=150, step=1)

    # Run button
    run_clustering = st.button('Run Clustering')

if run_clustering:
    st.session_state['run_clustering'] = True
    
    # Apply region filter if a specific region is selected; otherwise, use all data
    if selected_region != "All":
        filtered_df = df[(df['crashSeverity'].isin(selected_severities)) & (df['region'] == selected_region)]
    else:
        filtered_df = df[df['crashSeverity'].isin(selected_severities)]
    
    if not filtered_df.empty:
        # Perform clustering
        cluster_labels = perform_clustering(filtered_df, epsilon_km, min_samples)
        
        # Assign cluster labels directly to the DataFrame
        filtered_df['cluster'] = cluster_labels
        
        # Filter out noise (-1 labels) AFTER assigning cluster labels
        clustered_df = filtered_df[filtered_df['cluster'] != -1]
        
        # Update session state with the newly filtered and clustered DataFrame
        st.session_state['clustered_df'] = clustered_df
    else:
        # Handle case where filtering results in an empty DataFrame
        st.session_state['clustered_df'] = pd.DataFrame()
        st.warning("No data matches the selected filters. Please adjust your selections.")


main_col, right_sidebar = st.columns([9, 4])

with main_col:
    if st.session_state.get('run_clustering', False) and 'clustered_df' in st.session_state and st.session_state['clustered_df'] is not None and not st.session_state['clustered_df'].empty:
        
        col1, col2 = st.columns([1, 4])  # Adjust the ratio as needed to fit the select box nicely
        
        with col1:
            unique_clusters = np.sort(st.session_state['clustered_df']['cluster'].unique())
            selected_cluster = st.selectbox("Select a cluster to view on map:", unique_clusters, key='selected_cluster')
        
        if selected_cluster is not None:
            selected_cluster_df = st.session_state['clustered_df'][st.session_state['clustered_df']['cluster'] == selected_cluster]
            result_map = plot_on_map(selected_cluster_df)
            map_response = st_folium(result_map, width=650, height=500)

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
            st.markdown("**Summary of clusters:**")
            cluster_summary_df = prepare_cluster_summary(st.session_state['clustered_df'])
            st.dataframe(cluster_summary_df)  # Displaying the DataFrame for all clusters

with right_sidebar:

    # Display Cluster Summary
    if 'selected_cluster' in st.session_state and st.session_state['selected_cluster'] is not None:
        st.markdown("**Details for selected cluster:**")

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
                # Filter out columns where all values are either NULL or zero
                filtered_crash_df = selected_crash_detail_df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
                filtered_crash_df = filtered_crash_df.loc[:, (filtered_crash_df != 0).any(axis=0)]  # Further drop columns where all values are 0
                filtered_crash_df = filtered_crash_df.drop(['X'], axis=1)
                filtered_crash_df = filtered_crash_df.drop(['Y'], axis=1)

                # Assuming latitude and longitude are available for the selected crash
                crash_latitude = selected_crash_detail_df['latitude'].values[0]
                crash_longitude = selected_crash_detail_df['longitude'].values[0]


                # Generate the street view URL
                street_view_url = f"http://maps.google.com/maps?q=&layer=c&cbll={crash_latitude},{crash_longitude}"

                # Display the transposed DataFrame for better readability, showing only non-zero, non-NULL columns
                st.markdown("**Details for the selected crash:**")

                # Display the URL as a clickable hyperlink
                st.markdown(f"[View Crash Location]({street_view_url})", unsafe_allow_html=True)

                # Transpose the DataFrame, reset the index, and rename the columns to 'Field' and 'Value'
                transposed_df = filtered_crash_df.T.reset_index()
                transposed_df.columns = ['Field', 'Value']
                st.dataframe(transposed_df, hide_index=True)
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