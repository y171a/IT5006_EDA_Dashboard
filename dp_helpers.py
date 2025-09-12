import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans

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
    
    # Update layout with title at top and x-axis labels at bottom
    fig.update_layout(
        title=f'Missing Matrix (Sparse Visualization){sample_info}',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(missing_matrix.columns))),
            ticktext=[col for col in missing_matrix.columns],
            tickangle=-90,
            side='bottom',  # Changed from 'top' to 'bottom'
            showgrid=False,
            zeroline=False,
            range=[-1, len(missing_matrix.columns) + 7]
        ),
        yaxis=dict(
            showticklabels=False,
            autorange='reversed',
            showgrid=False,
            zeroline=False,
            range=[0, len(missing_matrix)]  # Adjusted range
        ),
        height=600,
        width=1200,
        plot_bgcolor='white',
        showlegend=False,
        margin=dict(l=50, r=100, t=80, b=120)  # Increased bottom margin for x-axis labels
    )
    
    # Add row count annotation
    fig.add_annotation(
        text=f"{len(missing_matrix)}",
        x=-1,
        y=len(missing_matrix)/2,
        showarrow=False,
        font=dict(size=12)
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

def detect_and_plot_low_variance_features(df, threshold=0.01, max_features=12):
    """Detects and visualizes zero and near-zero variance features."""
    
    # Exclude ID columns from analysis
    analysis_cols = [col for col in df.columns if col not in ID_COLUMNS]
    
    # Detect low variance features
    low_var_features = []
    for col in analysis_cols:
        freq = df[col].value_counts(normalize=True, dropna=False)
        if len(freq) == 1:
            low_var_features.append({
                'Feature': col, 
                'Type': 'Zero Variance',
                'Dominant Value': freq.index[0],
                'Dominant %': 100.0
            })
        elif freq.iloc[0] > (1 - threshold):
            low_var_features.append({
                'Feature': col, 
                'Type': 'Near-Zero Variance',
                'Dominant Value': freq.index[0],
                'Dominant %': freq.iloc[0] * 100
            })
    
    low_var_df = pd.DataFrame(low_var_features)
    
    if len(low_var_df) == 0:
        return None, None
    
    # Summary figure
    summary_fig = px.bar(
        low_var_df,
        x='Feature',
        y='Dominant %',
        color='Type',
        title=f'Low Variance Features (>{100*(1-threshold):.0f}% single value)',
        hover_data=['Dominant Value'],
        color_discrete_map={
            'Zero Variance': '#e74c3c',
            'Near-Zero Variance': '#f39c12'
        }
    )
    summary_fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        yaxis_title='Percentage of Most Common Value'
    )
    
    # Distribution plots
    features_to_plot = low_var_df['Feature'].head(max_features).tolist()
    
    # Create subplots
    from plotly.subplots import make_subplots
    n_features = len(features_to_plot)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=features_to_plot,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    for idx, feature in enumerate(features_to_plot):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        # Get value counts
        value_counts = df[feature].value_counts().head(10)  # Limit to top 10 values
        
        # Create bar chart
        fig.add_trace(
            go.Bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                name=feature,
                showlegend=False,
                hovertemplate='%{x}: %{y}<extra></extra>',
                marker_color='#3498db'
            ),
            row=row, col=col
        )
        
        # Update axes
        fig.update_xaxes(tickangle=-45, row=row, col=col)
        fig.update_yaxes(title_text='Count', row=row, col=col)
    
    fig.update_layout(
        title='Distribution of Low Variance Features',
        height=300 * n_rows,
        showlegend=False
    )
    
    return low_var_df, summary_fig, fig
