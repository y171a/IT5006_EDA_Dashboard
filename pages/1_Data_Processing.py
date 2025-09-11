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

# Dashboard Header with custom styling
st.markdown("""
<style>
    .dashboard-header {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dashboard-header">
    <h1>📊 Data Processing Dashboard</h1>
    <p>Interactive analysis of data quality, missing values, and processing decisions</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,4,1])
with col2:
    if st.button("🔄 Clear Cache and Reload Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
        
# Load data (cached)
df = load_data()

# Identifier columns
ID_COLUMNS = ['encounter_id', 'patient_nbr']

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
    # Exclude ID columns from correlation analysis
    analysis_cols = [col for col in df.columns if col not in ID_COLUMNS]
    missing_cols = [col for col in analysis_cols if df[col].isnull().any()]
    
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
        title='Missing Value Correlation Heatmap (excluding ID columns)',
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed',
        height=500
    )
    return fig

def plot_interactive_matrix(df, sample_size=None, max_cols=50):
    """Creates an interactive version of the missingno matrix plot using Plotly."""
    # Only sample if specified
    if sample_size and len(df) > sample_size:
        df_subset = df.sample(n=sample_size, random_state=42)
        sample_info = f" (Sample: {sample_size} rows)"
    else:
        df_subset = df
        sample_info = ""
    
    if df_subset.shape[1] > max_cols:
        missing_counts = df_subset.isnull().sum()
        top_missing_cols = missing_counts.nlargest(max_cols).index
        df_subset = df_subset[top_missing_cols]
    
    # Create binary matrix (0 for missing, 1 for present)
    missing_matrix = (~df_subset.isnull()).astype(int)
    
    # Create the main heatmap
    fig = go.Figure()
    
    # Main matrix plot
    fig.add_trace(go.Heatmap(
        z=missing_matrix.values,
        x=list(range(len(missing_matrix.columns))),
        y=list(range(len(missing_matrix))),
        colorscale=[[0, 'white'], [1, 'black']],
        showscale=False,
        xgap=1,
        ygap=0,
        hovertemplate='Row: %{y}<br>Column: %{customdata}<br>Present: %{z}<extra></extra>',
        customdata=[missing_matrix.columns.tolist()] * len(missing_matrix),
        name='Data Matrix'
    ))
    
    # Calculate row completeness for sparkline
    row_completeness = missing_matrix.sum(axis=1)
    
    # Add sparkline on the right
    sparkline_x = [len(missing_matrix.columns) + 1] * len(row_completeness)
    sparkline_width = row_completeness.values / len(missing_matrix.columns) * 5  # Scale for visibility
    
    # Add horizontal bars for sparkline
    for i, (x, width) in enumerate(zip(sparkline_x, sparkline_width)):
        fig.add_shape(
            type="rect",
            x0=x, x1=x + width,
            y0=i - 0.4, y1=i + 0.4,
            fillcolor="black",
            line_width=0,
        )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(missing_matrix.columns))),
            ticktext=[col for col in missing_matrix.columns],
            tickangle=-90,
            side='top',
            showgrid=False,
            zeroline=False,
            range=[-1, len(missing_matrix.columns) + 7]
        ),
        yaxis=dict(
            showticklabels=False,
            autorange='reversed',
            showgrid=False,
            zeroline=False,
            range=[-2, len(missing_matrix)]
        ),
        height=600,
        width=1200,
        plot_bgcolor='white',
        showlegend=False,
        margin=dict(l=50, r=100, t=120, b=80)
    )
    
    # Add text annotations
    fig.add_annotation(
        text=f"{len(missing_matrix)}",
        x=-1,
        y=len(missing_matrix)/2,
        showarrow=False,
        font=dict(size=12)
    )
    
    # Title at bottom
    fig.add_annotation(
        text=f"Missing Matrix (Sparse Visualization){sample_info}",
        x=len(missing_matrix.columns)/2,
        y=len(missing_matrix) + 1,
        showarrow=False,
        font=dict(size=14, family="Arial"),
        xanchor='center'
    )
    
    return fig

def plot_interactive_dendrogram(df, max_features=30):
    """Creates an interactive dendrogram showing the clustering of missing values."""
    # Exclude ID columns from clustering analysis
    analysis_cols = [col for col in df.columns if col not in ID_COLUMNS]
    missing_cols = [col for col in analysis_cols if df[col].isnull().any()]
    
    if len(missing_cols) < 2:
        fig = go.Figure()
        fig.update_layout(
            title="Not enough columns with missing data to generate a dendrogram",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig
    
    # Limit features if too many
    if len(missing_cols) > max_features:
        missing_percentages = df[missing_cols].isnull().mean()
        # Select columns with varied missing percentages
        selected_cols = missing_percentages.nlargest(max_features).index.tolist()
    else:
        selected_cols = missing_cols
    
    # Create binary matrix for missing values
    missing_matrix = df[selected_cols].isnull().astype(int)
    
    # Calculate correlation matrix
    corr_matrix = missing_matrix.corr()
    
    # Create distance matrix
    distance_matrix = 1 - corr_matrix.abs()
    
    # Perform hierarchical clustering
    condensed_dist = squareform(distance_matrix.values)
    Z = linkage(condensed_dist, method='average')
    
    # Create dendrogram
    dendro = dendrogram(Z, labels=selected_cols, no_plot=True)
    
    # Create the plot
    fig = go.Figure()
    
    # Add dendrogram lines
    for i, (xs, ys) in enumerate(zip(dendro['icoord'], dendro['dcoord'])):
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='lines',
            line=dict(color='black', width=1.5),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title='Missing Matrix (Hierarchical Clustering of Missingness) - ID columns excluded',
        xaxis=dict(
            tickmode='array',
            tickvals=[x*10 + 5 for x in range(len(selected_cols))],
            ticktext=dendro['ivl'],
            tickangle=-90,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            title='Distance'
        ),
        height=600,
        width=1200,
        plot_bgcolor='white',
        margin=dict(l=80, r=20, t=80, b=150)
    )
    
    return fig

def plot_missing_patterns_clustered(df, n_clusters=5):
    """Shows a simpler visualization of missing data patterns using clustering."""
    # Exclude ID columns from clustering
    analysis_cols = [col for col in df.columns if col not in ID_COLUMNS]
    missing_cols = [col for col in analysis_cols if df[col].isnull().any()]
    
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
        title=f'Columns Grouped by Missing Data Patterns ({n_clusters} clusters) - ID columns excluded',
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

# --- Missing Data Visualizations ---
left_col, right_col = st.columns([2, 1])
with left_col:
    # Missing data visualization container
    st.subheader("📊 Missing Data Patterns")
    
    # Tabs for different visualizations - ALL 5 TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Bar Chart", 
        "Heatmap", 
        "Matrix", 
        "Dendrogram",  # This was missing
        "Clusters"
    ])
    
    with tab1:
        fig_bar = plot_interactive_bar(df)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        fig_heatmap = plot_interactive_correlation_heatmap(df)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            use_sampling = st.checkbox("Use sampling for performance", value=True)
        with col_b:
            if use_sampling:
                sample_size = st.slider("Sample size", 100, 1000, 500, 100)
            else:
                sample_size = None
        
        fig_matrix = plot_interactive_matrix(df, sample_size=sample_size, max_cols=50)
        st.plotly_chart(fig_matrix, use_container_width=True)
    
    with tab4:  # DENDROGRAM TAB
        max_features = st.slider("Max features for dendrogram", min_value=10, max_value=50, value=25, step=5)
        fig_dendrogram = plot_interactive_dendrogram(df, max_features=max_features)
        st.plotly_chart(fig_dendrogram, use_container_width=True)
        st.info("**How to read:**\n- Columns connected at lower heights have similar missing patterns\n- Distance = 1 - |correlation| of missingness patterns")
    
    with tab5:  # CLUSTERS TAB (previously tab4)
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
        ⚠️ **Important: Dropping the 'weight' Column**

        The `weight` column will be removed from the dataset due to the following reasons:

        - **97% missing values**: With nearly 97% of data missing, any imputation would be highly speculative and likely introduce significant noise
        - **Exceeds practical threshold**: This level of missingness fails the threshold for meaningful statistical inference or predictive modeling
        - **Spurious clustering**: While it clusters with `payer_code` and `medical_specialty` in the dendrogram, this is driven by shared sparsity—not semantic or clinical relevance
        - **Limited clinical utility**: The extreme sparsity makes it unsuitable for reliable analysis or modeling

        **Decision**: Drop the column to maintain data integrity and model reliability.
        """)
        
        # Weight statistics
        weight_missing = df['weight'].isnull().sum()
        weight_missing_pct = (weight_missing / len(df)) * 100
        weight_present = (~df['weight'].isnull()).sum()
        df.drop(columns='weight', inplace=True)

# --- Feature Uniqueness Analysis ---
st.divider()
st.header("📊 Feature Data Uniqueness Checks")

# Calculate distinct counts
distinct_counts = pd.DataFrame({
    'Feature': df.columns,
    'Distinct Count': df.nunique().values
}).sort_values(by='Distinct Count', ascending=False)

# Display the dataframe
st.dataframe(distinct_counts, use_container_width=True)

# Note about ID columns
if any(col in df.columns for col in ID_COLUMNS):
    st.info(f"**Note**: Columns {ID_COLUMNS} are identifier columns with high uniqueness by design.")

# Create two columns for the two charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Chart WITH identifiers
    fig_with_ids = px.bar(
        distinct_counts.head(20),
        x='Feature',
        y='Distinct Count',
        title='Top 20 Features by Distinct Count (Including IDs)'
    )
    fig_with_ids.update_layout(
        xaxis_tickangle=-45,
        height=400
    )
    st.plotly_chart(fig_with_ids, use_container_width=True)

with chart_col2:
    # Chart WITHOUT identifiers
    distinct_counts_no_ids = distinct_counts[~distinct_counts['Feature'].isin(ID_COLUMNS)]
    
    fig_no_ids = px.bar(
        distinct_counts_no_ids.head(20),
        x='Feature',
        y='Distinct Count',
        title='Top 20 Features by Distinct Count (Excluding IDs)'
    )
    fig_no_ids.update_layout(
        xaxis_tickangle=-45,
        height=400
    )
    st.plotly_chart(fig_no_ids, use_container_width=True)

# Show features with very low uniqueness
low_uniqueness = distinct_counts[distinct_counts['Distinct Count'] <= 2]
if len(low_uniqueness) > 0:
    st.warning(f"**{len(low_uniqueness)} features** with ≤2 unique values")
    
    # Create scrollable container showing all features
    with st.container(height=120):
        for feat in low_uniqueness['Feature']:
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
