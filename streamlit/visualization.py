import streamlit.streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(layout="wide")

# Sample data
@st.cache_data
def load_data():
    return pd.DataFrame({
        'Category': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'Value': np.random.randint(10, 100, 100),
        'Sales': np.random.randint(100, 1000, 100),
        'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr'], 100)
    })

df = load_data()

st.title("Data Visualization Dashboard")

# Create columns
col1, col2 = st.columns(2)

# Matplotlib + Seaborn visualization
with col1:
    st.subheader("Seaborn Histogram")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x='Value', kde=True, ax=ax, color='skyblue')
    ax.set_title('Distribution of Values')
    st.pyplot(fig)

# Plotly visualization
with col2:
    st.subheader("Plotly Scatter Plot")
    fig_scatter = px.scatter(df, x='Value', y='Sales', color='Category', 
                            title='Value vs Sales', size='Value')
    st.plotly_chart(fig_scatter, use_container_width=True)

# Second row with more columns
col3, col4 = st.columns(2)

with col3:
    st.subheader("Bar Chart")
    fig_bar = px.bar(df.groupby('Category')['Sales'].sum().reset_index(), 
                    x='Category', y='Sales', title='Sales by Category')
    st.plotly_chart(fig_bar, use_container_width=True)

with col4:
    st.subheader("Box Plot")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Category', y='Value', ax=ax, palette='Set2')
    ax.set_title('Value Distribution by Category')
    st.pyplot(fig)