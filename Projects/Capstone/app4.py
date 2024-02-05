import streamlit as st
import pandas as pd
from sklearn.cluster import DBSCAN
import folium
from streamlit_folium import st_folium
import numpy as np

APP_TITLE = 'Fraud and Identity Theft Report'
APP_SUB_TITLE = 'Source: Federal Trade Commission'


# Define a function to load data
@st.cache_resource  # Adjusted cache decorator
def load_data(url):
    return pd.read_parquet(url)

# Define a function to perform clustering
def perform_clustering(subset_df, epsilon_km, min_samples):
    kms_per_radian = 6371.0088
    epsilon = epsilon_km / kms_per_radian
    coords = subset_df[['latitude', 'longitude']].values
    db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    return db.labels_

# Function to update the map with selected cluster's crashes
def update_map_with_selected_cluster(clustered_df, selected_cluster, center=(-41, 174), zoom_start=6):
    folium_map = folium.Map(location=center, zoom_start=zoom_start, tiles="cartodbpositron")
    severity_colors = {
        'Non-Injury Crash': '#1f77b4',
        'Minor Crash': '#ffec8b',
        'Serious Crash': '#d62728',
        'Fatal Crash': '#4d4d4d'
    }
    # Filter for the selected cluster
    df_selected_cluster = clustered_df[clustered_df['cluster'] == selected_cluster]
    for idx, row in df_selected_cluster.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=severity_colors[row['crashSeverity']],
            fill=True,
            fill_color=severity_colors[row['crashSeverity']],
            popup=f"Cluster: {row['cluster']}"
        ).add_to(folium_map)
    st_folium(folium_map, width=725, height=500)


# Main function encapsulating app logic
def main():
    st.title('Crash Data Clustering')

    # Load the DataFrame
    url = 'https://github.com/LouisMcArdell/IOD-Work/raw/main/Projects/Capstone/app_data.parquet'
    df = load_data(url)

    # UI elements for input
    severities = ['Non-Injury Crash', 'Minor Crash', 'Serious Crash', 'Fatal Crash']
    selected_severities = [severity for severity in severities if st.checkbox(severity, key=severity)]
    epsilon_km = st.number_input('Epsilon distance in km (e.g., 0.03 for 30m)', value=0.03, step=0.01)
    min_samples = st.number_input('Min samples', value=150, step=1)

    if st.button('Run Clustering'):
        filtered_df = df[df['crashSeverity'].isin(selected_severities)]
        if not filtered_df.empty:
            cluster_labels = perform_clustering(filtered_df, epsilon_km, min_samples)
            filtered_df['cluster'] = cluster_labels
            clustered_df = filtered_df[cluster_labels != -1].copy()
            st.session_state['clustered_df'] = clustered_df

        if 'clustered_df' in st.session_state:
            clustered_df = st.session_state['clustered_df']
            cluster_summary = clustered_df['cluster'].value_counts().sort_index()
            st.write("Cluster summary (Cluster Label: Count):")
            
            # Create a column layout for selection
            col1, col2 = st.columns([1, 4])
            
            with col1:
                # Generate a button for each cluster
                for cluster in cluster_summary.index:
                    if st.button(f"Select Cluster {cluster}"):
                        st.session_state['selected_cluster'] = cluster
            
            with col2:
                # Display the cluster summary table
                st.table(cluster_summary)

            # Check if a cluster has been selected and display the map
            if 'selected_cluster' in st.session_state and st.session_state['selected_cluster'] is not None:
                # Update the map for the selected cluster
                selected_cluster = st.session_state['selected_cluster']
                update_map_with_selected_cluster(clustered_df, selected_cluster)
                

if __name__ == "__main__":
    main()
