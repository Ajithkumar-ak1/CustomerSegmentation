"""Streamlit application for Customer Segmentation."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

from data_processor import DataProcessor
from kmeans_model import KMeansSegmentation

# Page config
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Customer Segmentation ML")
st.markdown("Segment customers using K-means clustering")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    n_clusters = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=5,
    )
    
    st.markdown("---")
    st.markdown("""
    ### About
    This app segments customers based on:
    - Annual Income
    - Spending Score
    """)

# File uploader
uploaded_file = st.file_uploader(
    "📁 Upload CSV file",
    type=["csv"],
    help="CSV with customer data"
)

if uploaded_file is not None:
    # Read file
    stringio = StringIO(uploaded_file.getvalue().decode("utf8"))
    df = pd.read_csv(stringio)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🔍 Elbow", "🎯 Clusters", "📈 Insights"])
    
    # TAB 1: Data Overview
    with tab1:
        st.header("Data Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing", df.isnull().sum().sum())
        
        st.markdown("### Dataset")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("### Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    
    # TAB 2: Elbow Method
    with tab2:
        st.header("Elbow Method - Find Optimal Clusters")
        
        with st.spinner("Calculating WCSS..."):
            processor = DataProcessor()
            processor.data = df
            features = processor.prepare_features([3, 4])
            
            model = KMeansSegmentation()
            k_values, wcss = model.find_optimal_clusters(features, k_range=(1, 11))
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(k_values, wcss, "bo-", linewidth=2, markersize=8)
            ax.set_xlabel("Number of Clusters", fontsize=12)
            ax.set_ylabel("WCSS", fontsize=12)
            ax.set_title("Elbow Method", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Table
        elbow_df = pd.DataFrame({"k": k_values, "WCSS": wcss})
        st.dataframe(elbow_df, use_container_width=True)
    
    # TAB 3: Clustering Results
    with tab3:
        st.header("Customer Segmentation")
        
        with st.spinner(f"Training with {n_clusters} clusters..."):
            processor = DataProcessor()
            processor.data = df
            features = processor.prepare_features([3, 4])
            
            model = KMeansSegmentation()
            labels = model.train(features, n_clusters=n_clusters)
            centers = model.get_cluster_centers()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Scatter plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
                     "#F7DC6F", "#BB8FCE", "#85C1E2", "#FDDB92", "#C1F0F6"]
            
            for i in np.unique(labels):
                mask = labels == i
                ax.scatter(
                    features[mask, 0],
                    features[mask, 1],
                    c=colors[i % len(colors)],
                    label=f"Cluster {i}",
                    s=60,
                    alpha=0.7,
                )
            
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                c="red",
                marker="X",
                s=300,
                label="Centroids",
                edgecolors="black",
                linewidth=2,
            )
            
            ax.set_xlabel("Annual Income (k$)", fontsize=12)
            ax.set_ylabel("Spending Score (1-100)", fontsize=12)
            ax.set_title("Customer Segments", fontsize=14, fontweight="bold")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            # Distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            cluster_sizes = [sum(labels == i) for i in np.unique(labels)]
            ax.bar(
                [f"C{i}" for i in np.unique(labels)],
                cluster_sizes,
                color=colors[:n_clusters],
                alpha=0.7,
            )
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title("Cluster Sizes", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")
            st.pyplot(fig)
        
        # Add cluster labels and download
        df_result = df.copy()
        df_result["Cluster"] = labels
        
        st.markdown("### Segmented Data")
        st.dataframe(df_result, use_container_width=True)
        
        csv = df_result.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name="customer_segments.csv",
            mime="text/csv"
        )
    
    # TAB 4: Insights
    with tab4:
        st.header("Cluster Insights")
        
        with st.spinner("Analyzing clusters..."):
            processor = DataProcessor()
            processor.data = df
            features = processor.prepare_features([3, 4])
            
            model = KMeansSegmentation()
            labels = model.train(features, n_clusters=n_clusters)
            cluster_info = model.get_cluster_info(features, labels)
        
        for cluster_name, info in cluster_info.items():
            with st.expander(f"📌 {cluster_name}", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Size", info["Size"])
                with col2:
                    st.metric("Percentage", info["Percentage"])
                with col3:
                    st.metric("Avg Income", info["Avg Income"])
                with col4:
                    st.metric("Avg Spending", info["Avg Spending"])

else:
    st.info("👆 Upload a CSV file to get started")
    
    st.markdown("""
    ### Expected CSV Format
    Your CSV should contain:
    - Column 3: Annual Income
    - Column 4: Spending Score
    
    Example columns: CustomerID, Gender, Age, Annual Income, Spending Score
    """)
