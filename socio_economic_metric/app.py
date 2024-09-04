import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib.colors import ListedColormap

# Set maximum width of the sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;  /* Adjust the width as needed */
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;  /* Adjust the width as needed */
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the scaler, cluster centers, and labels
scaler_path = 'scaler.pkl'
cluster_centers_path = 'cluster_centers.pkl'
cluster_labels_path = 'cluster_labels.pkl'

try:
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(cluster_centers_path, 'rb') as f:
        cluster_centers = pickle.load(f)
    with open(cluster_labels_path, 'rb') as f:
        cluster_labels = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}. Please ensure the file is in the correct directory.")
    st.stop()


# Function to predict the cluster based on new data
def predict_cluster(features):
    scaled_features = scaler.transform([features])
    distances = euclidean_distances(scaled_features, cluster_centers)
    closest_cluster = np.argmin(distances)
    return closest_cluster, distances[0], scaled_features[0]


# Title
st.title("Country Clustering Prediction")

# Sidebar input for features
st.sidebar.header("Input Features")

child_mort = st.sidebar.number_input('Child Mortality Rate', min_value=0.0, max_value=200.0, value=30.0, step=0.1)
exports = st.sidebar.number_input('Exports (% of GDP)', min_value=0.0, max_value=100.0, value=30.0, step=0.1)
health = st.sidebar.number_input('Health (% of GDP)', min_value=0.0, max_value=100.0, value=6.0, step=0.1)
imports = st.sidebar.number_input('Imports (% of GDP)', min_value=0.0, max_value=100.0, value=30.0, step=0.1)
income = st.sidebar.number_input('Income per capita', min_value=0.0, max_value=100000.0, value=10000.0, step=100.0)
inflation = st.sidebar.number_input('Inflation rate', min_value=-10.0, max_value=100.0, value=5.0, step=0.1)
life_expec = st.sidebar.number_input('Life Expectancy', min_value=20.0, max_value=100.0, value=70.0, step=0.1)
total_fer = st.sidebar.number_input('Total Fertility Rate', min_value=1.0, max_value=10.0, value=2.5, step=0.1)
gdpp = st.sidebar.number_input('GDP per capita', min_value=0.0, max_value=100000.0, value=5000.0, step=100.0)

# Organize the input features
input_features = [child_mort, exports, health, imports, income, inflation, life_expec, total_fer, gdpp]

# Button for prediction and plot
if st.sidebar.button('Predict Cluster'):
    cluster_label, distances, scaled_features = predict_cluster(input_features)
    st.success(f'The predicted cluster for the input features is: {cluster_label}')

    # Create a color map for clusters
    cmap = ListedColormap(['red', 'green', 'blue', 'orange', 'purple'])  # Adjust colors as needed

    # Scatter plot
    plt.figure(figsize=(10, 8))

    # Plot all cluster centers with different colors
    for i, center in enumerate(cluster_centers):
        plt.scatter(center[0], center[1], c=cmap(i), marker='x', s=100, label=f'Cluster {i}')

    # Plot the user's input features
    plt.scatter(scaled_features[0], scaled_features[1], c='cyan', marker='o', label='User Input', s=200)

    # Annotate the user input point
    plt.annotate('User Input', (scaled_features[0], scaled_features[1]), textcoords="offset points", xytext=(10, -10),
                 ha='center')

    # Draw lines connecting the user input to each cluster center
    for i, center in enumerate(cluster_centers):
        plt.plot([scaled_features[0], center[0]], [scaled_features[1], center[1]], 'k--')
        plt.text((scaled_features[0] + center[0]) / 2, (scaled_features[1] + center[1]) / 2,
                 f'Dist: {distances[i]:.2f}', fontsize=9, color='black')

    # Plotting the input feature with selected features (e.g., Child Mortality Rate and Exports)
    plt.xlabel('Child Mortality Rate (Scaled)')
    plt.ylabel('Exports (% of GDP) (Scaled)')
    plt.title('Scatter Plot of User Input and Cluster Centers')
    plt.legend()
    st.pyplot(plt)
