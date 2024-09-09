# Socio-Economic Metric Clustering
![Screenshot 2024-09-04 191044](https://github.com/user-attachments/assets/0d030ca7-4b78-43b0-955c-41598177ed66)
 
## Overview
This project analyzes socio-economic data from various countries to perform clustering based on several metrics. The analysis includes exploratory data analysis (EDA), feature scaling, and clustering using Agglomerative Hierarchical Clustering. The final results are visualized, and a Streamlit app is provided to allow users to predict clusters for new data.

## Project Structure
- data/: Contains the dataset used for analysis.
- notebooks/: Jupyter notebooks with the code for data analysis, EDA, and clustering.
- app/: Contains the Streamlit application code for interactive clustering predictions.
- scaler.pkl: Serialized StandardScaler object for feature scaling.
- cluster_centers.pkl: Serialized cluster centers obtained from clustering.
- cluster_labels.pkl: Serialized cluster labels for each data point. 
## Features
- Exploratory Data Analysis (EDA): Visualizations of key socio-economic metrics such as Child Mortality Rate, Fertility Rate, Life Expectancy, Health Spending, etc.
- Clustering: Agglomerative Hierarchical Clustering to group countries into clusters based on socio-economic features.
- Streamlit App: An interactive web application that allows users to input socio-economic metrics and predict which cluster the data belongs to.

## Usage
- Open the Streamlit app in your browser https://socio-economic-metric-ktmasdxxxudxru75kbfnua.streamlit.app/.
- Enter the socio-economic metrics in the sidebar.
- Click the "Predict Cluster" button to see the predicted cluster and visualize the results.
## Visualizations
- EDA: Includes bar plots and scatter plots to visualize the distribution of various socio-economic metrics across countries.
- Clustering Results: Scatter plots to visualize the clustering results, including the distance from user inputs to cluster centers.
 
