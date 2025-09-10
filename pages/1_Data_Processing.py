import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Hello import load_data
import missingno as msno
import plotly.express as px 
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans

st.set_page_config(page_title="Data Processing Dashboard", page_icon="⚙️", layout="wide")

# Dashboard Header
st.markdown("""
<div class="dashboard-header">
    <h1>📊 Data Processing Dashboard</h1>
    <p>Interactive analysis of data quality, missing values, and processing decisions</p>
</div>
""", unsafe_allow_html=True)

# Load data (cached)
df = load_data()

# --- Helper Functions ---
def plot_interactive_bar(df):
    """Recreates the missingno bar chart using Plotly for interactivity."""
    non_null_counts = df.notna().sum()
    total_records = len(df)
    plot_df = pd.DataFrame({'Column': non_null_counts.index, 'Records Present': non_null_counts.values})
    plot_df['Completeness'] = plot_df['Records Present'] / total_records
    fig = px.bar(plot_df, x='Column', y='Records Present', title="Data Completeness by Column",
                 labels={'Column': 'Feature', 'Records Present': 'Number of Non-Null Records'},
                 hover_data={'Completeness': ':.2%'})
    fig.update_layout(xaxis_tickangle=-45, height=500)
    return fig

def plot_interactive_correlation_heatmap(df):
    """Recreates the triangular, annotated missingno correlation heatmap using Plotly."""
    missing_cols = df.columns[df.isnull().any()].tolist()
    if len(missing_cols) < 2:
        fig = go.Figure()
        fig.update_layout(
            title="Not enough columns with missing data to generate a heatmap.",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig
    
    corr_matrix = df[missing_cols].isna().corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_masked = corr_matrix.mask(mask)
    
    z_values = corr_masked.values
    text_values = np.where(np.isnan(z_values), "", np.round(z_values, 1).astype(str))
    z_values = np.where(np.isnan(z_values), None, z_values)

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=corr_masked.columns,
        y=corr_masked.columns,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=text_values,
        texttemplate="%{text}",
        hovertemplate='Correlation: %{z:.2f}<br>%{y} vs %{x}<extra></extra>',
        showscale=True
    ))

    fig.update_layout(
        title='Missing Value Correlation Heatmap',
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed',
        height=500
    )
    return fig

def plot_interactive_matrix(df, sample_size=1000, max_cols=50):
    """Creates an interactive version of the missingno matrix plot using Plotly."""
    if len(df) > sample_size:
        df_subset = df.sample(n=sample_size, random_state=42)
        sample_info = f" (Sample: {sample_size} rows)"
    else:
        df_subset = df
        sample_info = ""
    
    if df_subset.shape[1] > max_cols:
        missing_counts = df_subset.isnull().sum()
        top_missing_cols = missing_counts.nlargest(max_cols).index
        df_subset = df_subset[top_missing_cols]
    
    missing_matrix = (~df_subset.isnull()).astype(int)
    
    fig = go.Figure(data=go.Heatmap(
        z=missing_matrix.values,
        x=missing_matrix.columns,
        y=list(range(len(missing_matrix))),
        colorscale=[[0, 'white'], [1, 'black']],
        showscale=False,
        hovertemplate='Row: %{y}<br>Column: %{x}<br>Value: %{z}<extra></extra>'
    ))
    
    col_completeness = (missing_matrix.sum() / len(missing_matrix) * 100).values
    
    fig.add_trace(go.Bar(
        x=missing_matrix.columns,
        y=col_completeness,
        yaxis='y2',
        marker_color='lightblue',
        name='Completeness %',
        hovertemplate='%{x}<br>Completeness: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Missing Data Matrix{sample_info}',
        xaxis=dict(tickangle=-45, side='top', showgrid=False),
        yaxis=dict(showticklabels=False, autorange='reversed', showgrid=False, domain=[0.2, 1]),
        yaxis2=dict(side='right', overlaying='y', range=[0, 100], showgrid=True, 
                    title='Completeness %', domain=[0, 0.15]),
        height=600,
        plot_bgcolor='white',
        showlegend=False
    )
    return fig

def plot_missing_patterns_clustered(df, n_clusters=5):
    """Shows a simpler visualization of missing data patterns using clustering."""
    missing_cols = df.columns[df.isnull().any()].tolist()
    
    if len(missing_cols) < n_clusters:
        n_clusters = max(2, len(missing_cols) - 1)
    
    missing_matrix = df[missing_cols].isnull().astype(int).T
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(missing_matrix)
    
    cluster_df = pd.DataFrame({
        'Column': missing_cols,
        'Cluster': clusters,
        'Missing %': df[missing_cols].isnull().mean() * 100
    }).sort_values(['Cluster', 'Missing %'])
    
    fig = px.bar(
        cluster_df, 
        x='Column', 
        y='Missing %',
        color='Cluster',
        title=f'Columns Grouped by Missing Data Patterns ({n_clusters} clusters)',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(height=500)
    
    return fig

# --- Top-level Metrics ---
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Dataset Rows", f"{len(df):,}")
with col2:
    st.metric("Features", f"{len(df.columns)}")
with col3:
    total_missing = df.isnull().sum().sum()
    st.metric("Missing Cells", f"{total_missing:,}")
with col4:
    missing_pct = (total_missing / (len(df) * len(df.columns))) * 100
    st.metric("Missing %", f"{missing_pct:.1f}%")
with col5:
    cols_with_missing = df.columns[df.isnull().any()].shape[0]
    st.metric("Columns w/ Missing", cols_with_missing)

st.markdown("---")

# --- Main Dashboard Layout ---
# Create two main columns
left_col, right_col = st.columns([2, 1])

with left_col:
    # Missing data visualization container
    st.subheader("📊 Missing Data Patterns")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Bar Chart", "Heatmap", "Matrix", "Clusters"])
    
    with tab1:
        fig_bar = plot_interactive_bar(df)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        fig_heatmap = plot_interactive_correlation_heatmap(df)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            sample_size = st.slider("Sample size", 100, 2000, 500, 100, key="matrix_sample")
        with col_b:
            max_cols = st.slider("Max columns", 10, 50, 30, 5, key="matrix_cols")
        fig_matrix = plot_interactive_matrix(df, sample_size, max_cols)
        st.plotly_chart(fig_matrix, use_container_width=True)
    
    with tab4:
        n_clusters = st.slider("Number of clusters", 2, 10, 5, key="cluster_n")
        fig_clusters = plot_missing_patterns_clustered(df, n_clusters)
        st.plotly_chart(fig_clusters, use_container_width=True)

with right_col:
    # Missing values summary
    st.subheader("🔍 Top Missing Values")
    
    missing_summary = pd.DataFrame({
        'Column': df.columns,
        'Missing %': df.isnull().mean() * 100,
        'Missing Count': df.isnull().sum()
    }).sort_values(by='Missing %', ascending=False).head(15)
    
    # Create a compact bar chart
    fig_summary = px.bar(
        missing_summary, 
        y='Column', 
        x='Missing %',
        orientation='h',
        title="Top 15 Columns by Missing %",
        height=600,
        hover_data={'Missing Count': ':,'}
    )
    fig_summary.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_summary, use_container_width=True)

# --- Detailed Analysis Section ---
st.markdown("---")
st.subheader("📋 Detailed Analyses")

# Use columns for the expandable sections
col_left, col_right = st.columns(2)

with col_left:
    with st.expander("🧪 A1C vs. Glucose Test Analysis", expanded=False):
        # Data preparation
        df_analysis = df.copy()
        df_analysis['max_glu_serum'] = df_analysis['max_glu_serum'].fillna("Not Measured")
        df_analysis['A1Cresult'] = df_analysis['A1Cresult'].fillna("Not Measured")
        
        df1 = df_analysis[~((df_analysis['max_glu_serum'] == 'Not Measured') & 
                           (df_analysis['A1Cresult'] == 'Not Measured'))].copy()
        
        df1['max_glu_serum'] = df1['max_glu_serum'].apply(lambda x: x if x == "Not Measured" else 'Measured')
        df1['A1Cresult'] = df1['A1Cresult'].apply(lambda x: x if x == "Not Measured" else 'Measured')
        
        # Calculate statistics
        glu_counts = df1['max_glu_serum'].value_counts()
        glu_pct = df1['max_glu_serum'].value_counts(normalize=True) * 100
        a1c_counts = df1['A1Cresult'].value_counts()
        a1c_pct = df1['A1Cresult'].value_counts(normalize=True) * 100
        
        # Create plot data
                # Create plot data
        plot_data = pd.DataFrame({
            'Test Type': ['Max Glucose Serum', 'Max Glucose Serum', 'A1C Result', 'A1C Result'],
            'Status': ['Measured', 'Not Measured', 'Measured', 'Not Measured'],
            'Count': [
                glu_counts.get('Measured', 0),
                glu_counts.get('Not Measured', 0),
                a1c_counts.get('Measured', 0),
                a1c_counts.get('Not Measured', 0)
            ],
            'Percentage': [
                glu_pct.get('Measured', 0),
                glu_pct.get('Not Measured', 0),
                a1c_pct.get('Measured', 0),
                a1c_pct.get('Not Measured', 0)
            ]
        })
        
        # Create interactive stacked bar chart
        fig_stacked = px.bar(
            plot_data,
            x='Test Type',
            y='Percentage',
            color='Status',
            title='Test Measurement Status',
            color_discrete_map={'Measured': '#2ecc71', 'Not Measured': '#e74c3c'},
            hover_data={'Count': ':,', 'Percentage': ':.1f'}
        )
        
        fig_stacked.update_traces(
            hovertemplate='<b>%{x}</b><br>' +
                          'Status: %{fullData.name}<br>' +
                          'Count: %{customdata[0]:,}<br>' +
                          'Percentage: %{y:.1f}%<br>' +
                          '<extra></extra>'
        )
        
        fig_stacked.update_layout(
            yaxis_title='Percentage (%)',
            xaxis_title='Test Type',
            hovermode='x unified',
            legend_title_text='Status',
            height=400
        )
        
        st.plotly_chart(fig_stacked, use_container_width=True)
        
        # Summary metrics
        both_measured = len(df1[(df1['max_glu_serum'] == 'Measured') & (df1['A1Cresult'] == 'Measured')])
        only_one = len(df1[(df1['max_glu_serum'] == 'Measured') ^ (df1['A1Cresult'] == 'Measured')])
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Both Tests Measured", f"{both_measured:,}")
        with metric_col2:
            st.metric("Only One Test", f"{only_one:,}")
        
        # Add crosstab table
        crosstab = pd.crosstab(df1["max_glu_serum"], df1["A1Cresult"])
        st.dataframe(crosstab)

with col_right:
    with st.expander("🔧 Data Cleaning: Weight Column", expanded=False):
        st.warning("""
        **Dropping 'weight' Column**
        
        - **97% missing values**: Imputation would be speculative
        - **Fails threshold**: Unsuitable for modeling
        - **Spurious clustering**: Correlation driven by sparsity, not clinical relevance
        """)
        
        # Weight statistics
        weight_missing = df['weight'].isnull().sum()
        weight_missing_pct = (weight_missing / len(df)) * 100
        weight_present = (~df['weight'].isnull()).sum()
        df.drop(columns='weight', inplace=True)
    

# --- Feature Uniqueness Analysis ---
st.markdown("---")
st.subheader("📊 Feature Data Uniqueness")

# Calculate distinct counts
distinct_counts = pd.DataFrame({
    'Feature': df.columns,
    'Distinct Count': df.nunique().values,
    'Uniqueness Ratio': (df.nunique() / len(df) * 100).round(2)
}).sort_values(by='Distinct Count', ascending=False)

# Create two columns for the uniqueness analysis
unique_col1, unique_col2 = st.columns([3, 2])

with unique_col1:
    # Interactive bar chart
    fig_uniqueness = px.bar(
        distinct_counts.head(20),
        x='Feature',
        y='Distinct Count',
        title='Top 20 Features by Distinct Count',
        hover_data=['Uniqueness Ratio'],
        color='Uniqueness Ratio',
        color_continuous_scale='Viridis'
    )
    
    fig_uniqueness.update_layout(
        xaxis_tickangle=-45,
        height=400,
        coloraxis_colorbar_title="Uniqueness %"
    )
    
    st.plotly_chart(fig_uniqueness, use_container_width=True)

with unique_col2:
    st.markdown("##### Key Insights")
    
    # High uniqueness features
    high_uniqueness = distinct_counts[distinct_counts['Uniqueness Ratio'] > 95]
    if len(high_uniqueness) > 0:
        st.info(f"**{len(high_uniqueness)} features** with >95% unique values")
        for feat in high_uniqueness['Feature'].head(5):
            st.text(f"• {feat}")
    
    # Low uniqueness features
    low_uniqueness = distinct_counts[distinct_counts['Distinct Count'] <= 2]
    if len(low_uniqueness) > 0:
        st.warning(f"**{len(low_uniqueness)} features** with ≤2 unique values")
        for feat in low_uniqueness['Feature'].head(5):
            st.text(f"• {feat}")

# --- Data Preview Section ---
st.markdown("---")
with st.expander("📄 View Full Dataset", expanded=False):
    # Add search/filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        search_col = st.selectbox("Search column:", ["All"] + list(df.columns))
    with col2:
        if search_col != "All":
            unique_vals = df[search_col].dropna().unique()
            if len(unique_vals) < 100:
                search_val = st.selectbox("Filter by value:", ["All"] + list(unique_vals))
            else:
                search_val = st.text_input("Filter value:")
        else:
            search_val = "All"
    with col3:
        show_rows = st.number_input("Show rows:", min_value=10, max_value=1000, value=100)
    
    # Apply filters
    df_display = df.head(show_rows)
    if search_col != "All" and search_val != "All" and search_val != "":
        df_display = df[df[search_col] == search_val].head(show_rows)
    
    st.dataframe(df_display, use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Dataset as CSV",
        data=csv,
        file_name='processed_data.csv',
        mime='text/csv'
    )

# --- Footer ---
st.markdown("---")
st.caption("Data Processing Dashboard | Last updated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))