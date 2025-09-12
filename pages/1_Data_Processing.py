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
    <h1>⚙️ Data Processing Dashboard</h1>
    <p>Interactive analysis of data quality, missing values, and processing decisions</p>
</div>
""", unsafe_allow_html=True)

# Load data helpers
from dp_helpers import (
    plot_interactive_bar,
    plot_interactive_correlation_heatmap,
    plot_interactive_matrix,
    plot_interactive_dendrogram,
    plot_missing_patterns_clustered
)

# Load data (cached)
df = load_data()

# Identifier columns
ID_COLUMNS = ['encounter_id', 'patient_nbr']

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
                sample_size = st.slider("Sample size", 100, 500, 500, 100)
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
low_uniqueness = distinct_counts[distinct_counts['Distinct Count'] <= 1]
if len(low_uniqueness) > 0:
    st.warning(f"**{len(low_uniqueness)} features** with ≤1 unique values")
    
    # Check if examide and citoglipton are in the low uniqueness features
    constant_drugs = ['examide', 'citoglipton']
    found_constants = [feat for feat in constant_drugs if feat in low_uniqueness['Feature'].values]
    other_constants = [feat for feat in low_uniqueness['Feature'] if feat not in constant_drugs]
    
    if found_constants:
        # Create two columns for constant drug analysis
        const_col1, const_col2 = st.columns([1, 2])
        
        with const_col1:
            st.markdown("**Constant Drug Features:**")
            # Create a styled container for the constant features
            with st.container():
                for feat in found_constants:
                    unique_val = df[feat].dropna().unique()
                    if len(unique_val) > 0:
                        st.markdown(f"• **{feat}**: Always '{unique_val[0]}'")
                    else:
                        st.markdown(f"• **{feat}**: No non-null values")
        
        with const_col2:
            with st.expander("🔧 Data Cleaning: Constant Drug Variables", expanded=False):
                st.warning("""
                ⚠️ **Important: Dropping the 'examide' and 'citoglipton' Columns**

                The `examide` and `citoglipton` columns will be removed from the dataset due to the following reasons:

                - **Zero variance**: These features have the same value for all records, providing no discriminatory power
                - **No predictive value**: Constants cannot help distinguish between different outcomes or patient groups
                - **Redundant information**: They add no information content to the model
                - **Computational efficiency**: Removing them reduces dimensionality without information loss
                - **Algorithm compatibility**: May cause issues with certain algorithms that require variance

                **Decision**: Drop these columns before modeling as they do not explain any variance in the target variable and cannot improve model performance.
                """)
                df.drop(columns=found_constants, inplace=True)

st.divider()
st.header("🔍 Low Variance Feature Analysis")

# Detect low variance features (fixed threshold)
NZV_THRESHOLD = 0.01
low_var_features = []

# Exclude ID columns
analysis_cols = [col for col in df.columns if col not in ID_COLUMNS]

for col in analysis_cols:
    freq = df[col].value_counts(normalize=True, dropna=False)
    if len(freq) == 1:
        low_var_features.append({
            'Feature': col, 
            'Type': 'Zero Variance',
            'Dominant Value': str(freq.index[0]),
            'Dominant %': 100.0,
            'Unique Values': 1
        })
    elif freq.iloc[0] > 0.99:  # 99% threshold
        low_var_features.append({
            'Feature': col, 
            'Type': 'Near-Zero Variance',
            'Dominant Value': str(freq.index[0]),
            'Dominant %': freq.iloc[0] * 100,
            'Unique Values': len(freq)
        })

if low_var_features:
    low_var_df = pd.DataFrame(low_var_features)
    
    # Simple summary
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display table
        st.dataframe(
            low_var_df.style.format({'Dominant %': '{:.1f}%'}).background_gradient(
                subset=['Dominant %'], 
                cmap='Reds', 
                vmin=99, 
                vmax=100
            ),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        with st.expander("🔧 Data Cleaning: Near-Zero Variance Medications", expanded=False):
            st.warning("""
            ⚠️ **Important: Dropping Near-Zero Variance Medication Columns**

            Multiple medication columns show >99% of patients with "No" values:

            - **Clinical interpretation**: These drugs were either not prescribed during the study period or are clinically irrelevant for the majority of diabetic cases
            - **Statistical limitation**: With near-zero variance, these features cannot explain variance in the target variable
            - **Predictive power**: Features dominated by a single value (>99%) cannot be meaningful predictors for readmission risk
            - **Model efficiency**: Removing these features reduces dimensionality without losing predictive capability

            **Decision**: Drop medication columns with >99% "No" values to improve model interpretability and computational efficiency.
            """)

# Drop low variance features
df.drop(columns=low_var_df.Feature.to_list(), inplace=True)

# --- Store Cleaned Dataset ---
# This should be at the very bottom of your data processing page
# after all the drop operations have been performed

# Store the cleaned dataframe in session state
st.session_state.cleaned_df = df

# --- Footer ---
st.markdown("---")
st.caption("Data Processing Dashboard | Last updated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
