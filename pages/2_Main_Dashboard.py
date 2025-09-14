import streamlit as st
import pandas as pd
from data_processing import get_processed_dataframe
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import skew

st.set_page_config(page_title="Main Dashboard", page_icon="📊", layout="wide")

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
    <h1>📊 Main Dashboard</h1>
    <p>Interactive analysis of main data features</p>
</div>
""", unsafe_allow_html=True)

# Load the processed data
df = get_processed_dataframe()

# Define color palette for readmission status
READMISSION_COLORS = {
    '<30': '#e74c3c',  # Red for urgent readmission
    '>30': '#f39c12',  # Orange for delayed readmission
    'NO': '#27ae60'    # Green for no readmission
}

# Add this after your header
st.markdown("---")
col1, col2, col3, col4 = st.columns([1, 3, 3, 3])
with col1:
    st.markdown("**Legend:**")
with col2:
    st.markdown(f"<span style='color: {READMISSION_COLORS['<30']}'>●</span> **<30**: Urgent readmission", unsafe_allow_html=True)
with col3:
    st.markdown(f"<span style='color: {READMISSION_COLORS['>30']}'>●</span> **>30**: Delayed readmission", unsafe_allow_html=True)
with col4:
    st.markdown(f"<span style='color: {READMISSION_COLORS['NO']}'>●</span> **NO**: No readmission", unsafe_allow_html=True)
st.markdown("---")

# Helper Functions
def get_repeated_patients_data(df):
    """Filter and return data for repeated patients."""
    df = pd.DataFrame(df)
    repeated_patients = df.groupby('patient_nbr').size().sort_values(ascending=False)
    repeated_ids = repeated_patients[repeated_patients > 1].index.tolist()
    df_repeated = df[df['patient_nbr'].isin(repeated_ids)].copy()
    return df_repeated, repeated_patients

def choose_plot_type(data, feature, target='readmitted', skew_threshold=1.0, min_samples=100):
    """Determine the best plot type for a numerical feature."""
    feature_data = data[[feature, target]].dropna()
    n = feature_data.shape[0]
    feature_skew = abs(skew(feature_data[feature]))
    
    if n < min_samples:
        return 'boxplot'
    elif feature_skew > skew_threshold:
        return 'distribution'
    else:
        counts, bins = np.histogram(feature_data[feature], bins=20)
        peaks = sum((counts[i] > counts[i-1]) and (counts[i] > counts[i+1]) 
                   for i in range(1, len(counts)-1))
        return 'distribution' if peaks > 1 else 'boxplot'

# Get repeated patients data
df_repeated, repeated_counts = get_repeated_patients_data(df)

# Sidebar for filtering
with st.sidebar:
    st.header("🎯 Filters & Settings")
    
    # Readmission filter
    readmission_filter = st.multiselect(
        "Select Readmission Status",
        options=['<30', '>30', 'NO'],
        default=['<30', '>30', 'NO']
    )
    
    # Top patients slider
    n_top_patients = st.slider(
        "Number of top patients to show",
        min_value=10,
        max_value=100,
        value=50,
        step=10
    )
    
    # Feature selection
    exclude_cols = {"readmitted", "encounter_id", "patient_nbr", 
                "diag_1", "diag_2", "diag_3",
                "discharge_disposition_id", "admission_type_id", "admission_source_id"}
    feature_cols = [col for col in df_repeated.columns if col not in exclude_cols]
    
    selected_features = st.multiselect(
        "Select Features to Analyze",
        options=feature_cols,
        default=feature_cols[:6]  # Show first 6 by default
    )

# Apply filters
df_filtered = df_repeated[df_repeated['readmitted'].isin(readmission_filter)].copy()

# Main Dashboard
# Top-level metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Repeated Patients", f"{len(repeated_counts[repeated_counts > 1]):,}")
with col2:
    st.metric("Total Encounters (Repeated)", f"{df_repeated.shape[0]:,}")
with col3:
    avg_encounters = repeated_counts[repeated_counts > 1].mean()
    st.metric("Avg Encounters per Patient", f"{avg_encounters:.1f}")
with col4:
    readmit_rate = (df_repeated['readmitted'] != 'NO').mean() * 100
    st.metric("Readmission Rate", f"{readmit_rate:.1f}%")

st.markdown("---")

# Two main sections
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("📊 Feature Distributions by Readmission Status")
    
    # Tabs for different view options
    tab1, tab2 = st.tabs(["Grid View", "Individual Features"])
    
    with tab1:
        if selected_features:
            # Create subplots grid
            n_features = len(selected_features)
            n_cols = 2
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=selected_features,
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            for idx, feature in enumerate(selected_features):
                row = idx // n_cols + 1
                col = idx % n_cols + 1
                
                if pd.api.types.is_numeric_dtype(df_filtered[feature]):
                    # Numerical feature
                    plot_type = choose_plot_type(df_filtered, feature)
                    
                    if plot_type == 'boxplot':
                        for readmit_status in df_filtered['readmitted'].unique():
                            data_subset = df_filtered[df_filtered['readmitted'] == readmit_status][feature]
                            fig.add_trace(
                                go.Box(
                                    y=data_subset,
                                    name=readmit_status,
                                    marker_color=READMISSION_COLORS[readmit_status],
                                    showlegend=(idx == 0)
                                ),
                                row=row, col=col
                            )
                    else:
                        for readmit_status in df_filtered['readmitted'].unique():
                            data_subset = df_filtered[df_filtered['readmitted'] == readmit_status][feature]
                            fig.add_trace(
                                go.Histogram(
                                    x=data_subset,
                                    name=readmit_status,
                                    opacity=0.7,
                                    marker_color=READMISSION_COLORS[readmit_status],
                                    histnorm='probability density',
                                    showlegend=(idx == 0)
                                ),
                                row=row, col=col
                            )
                else:
                    # Categorical feature
                    if df_filtered[feature].nunique() <= 20:
                        # Calculate conditional probabilities - simpler version
                        prob_df = (
                            df_filtered[df_filtered[feature].notna()]  # Filter NaN first
                            .groupby([feature, 'readmitted'])
                            .size()
                            .reset_index(name='count')
                        )
                        
                        # Calculate probabilities
                        total_by_feature = prob_df.groupby(feature)['count'].transform('sum')
                        prob_df['probability'] = prob_df['count'] / total_by_feature
                        # Create stacked bar chart
                        for readmit_status in prob_df['readmitted'].unique():
                            data_subset = prob_df[prob_df['readmitted'] == readmit_status]
                            fig.add_trace(
                                go.Bar(
                                    x=data_subset[feature],
                                    y=data_subset['probability'],
                                    name=readmit_status,
                                    marker_color=READMISSION_COLORS[readmit_status],
                                    showlegend=(idx == 0),
                                    text=data_subset['probability'].round(2),
                                    textposition='inside'
                                ),
                                row=row, col=col
                            )
            
            fig.update_layout(
                height=400 * n_rows,
                showlegend=True,
                barmode='stack',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select features from the sidebar to analyze.")
    
    with tab2:
        if selected_features:
            # Individual feature analysis with more details
            feature_to_analyze = st.selectbox("Select a feature for detailed view:", selected_features)
            
            if pd.api.types.is_numeric_dtype(df_filtered[feature_to_analyze]):
                # Create box plot with statistics
                fig = px.box(
                    df_filtered,
                    x='readmitted',
                    y=feature_to_analyze,
                    color='readmitted',
                    color_discrete_map=READMISSION_COLORS,
                    title=f"{feature_to_analyze} Distribution by Readmission Status",
                    points="outliers"  # Show outlier points
                )
                
                # Update hover template to include mean and other statistics
                for i, status in enumerate(['<30', '>30', 'NO']):  # Ensure consistent order
                    if status in df_filtered['readmitted'].unique():
                        # Calculate statistics for this group
                        group_data = df_filtered[df_filtered['readmitted'] == status][feature_to_analyze].dropna()
                        mean_val = group_data.mean()
                        median_val = group_data.median()
                        std_val = group_data.std()
                        
                        # Find the trace for this status
                        for trace in fig.data:
                            if trace.name == status:
                                trace.hovertemplate = (
                                    f"<b>{status}</b><br>" +
                                    f"Value: %{{y}}<br>" +
                                    f"<br><b>Statistics:</b><br>" +
                                    f"Mean: {mean_val:.2f}<br>" +
                                    f"Median: {median_val:.2f}<br>" +
                                    f"Std Dev: {std_val:.2f}<br>" +
                                    f"<extra></extra>"
                                )
                
                # Update layout for better appearance
                fig.update_layout(
                    yaxis_title=feature_to_analyze,
                    xaxis_title="Readmission Status",
                    height=500,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics table
                st.markdown("**Summary Statistics:**")
                stats_df = df_filtered.groupby('readmitted')[feature_to_analyze].agg([
                    'count', 'mean', 'median', 'std', 'min', 'max'
                ]).round(2)
                
                # Style the statistics table
                st.dataframe(
                    stats_df.style.format({
                        'count': '{:.0f}',
                        'mean': '{:.2f}',
                        'median': '{:.2f}',
                        'std': '{:.2f}',
                        'min': '{:.2f}',
                        'max': '{:.2f}'
                    }).background_gradient(cmap='RdYlGn_r', subset=['mean']),
                    use_container_width=True
                )
                
            else:
                # Categorical feature analysis
                # Create sunburst chart for categorical
                freq_df = df_filtered.groupby([feature_to_analyze, 'readmitted']).size().reset_index(name='count')
                
                # Calculate percentages within each category
                total_by_feature = freq_df.groupby(feature_to_analyze)['count'].transform('sum')
                freq_df['percentage'] = (freq_df['count'] / total_by_feature * 100).round(1)
                
                fig = px.sunburst(
                    freq_df,
                    path=[feature_to_analyze, 'readmitted'],
                    values='count',
                    color='readmitted',
                    color_discrete_map=READMISSION_COLORS,
                    title=f"{feature_to_analyze} Distribution by Readmission Status",
                    hover_data={'percentage': True}
                )
                
                # Update hover template
                fig.update_traces(
                    hovertemplate='<b>%{label}</b><br>' +
                                'Count: %{value}<br>' +
                                'Percentage: %{customdata[0]:.1f}%<br>' +
                                '<extra></extra>'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show frequency table
                st.markdown("**Frequency Table:**")
                
                # Create a pivot table for better readability
                pivot_df = freq_df.pivot_table(
                    index=feature_to_analyze,
                    columns='readmitted',
                    values='count',
                    fill_value=0
                )
                
                # Add percentage columns
                for col in pivot_df.columns:
                    pivot_df[f'{col}_pct'] = (pivot_df[col] / pivot_df.sum(axis=1) * 100).round(1)
                
                # Reorder columns for better display
                cols_ordered = []
                for status in ['<30', '>30', 'NO']:
                    if status in pivot_df.columns:
                        cols_ordered.extend([status, f'{status}_pct'])
                
                pivot_df = pivot_df[cols_ordered]
                
                # Style the table
                st.dataframe(
                    pivot_df.style.format({
                        col: '{:.0f}' if '_pct' not in col else '{:.1f}%' 
                        for col in pivot_df.columns
                    }).background_gradient(cmap='YlOrRd', subset=[col for col in pivot_df.columns if '_pct' not in col]),
                    use_container_width=True
                )
                
                # Add insights
                st.markdown("**Key Insights:**")
                
                # Find category with highest readmission rate
                readmit_rates = df_filtered.groupby(feature_to_analyze).apply(
                    lambda x: (x['readmitted'] != 'NO').mean() * 100
                ).round(1)
                
                highest_readmit_cat = readmit_rates.idxmax()
                highest_readmit_rate = readmit_rates.max()
                
                st.info(f"📊 '{highest_readmit_cat}' has the highest readmission rate at {highest_readmit_rate}%")

with right_col:
    st.subheader("👥 Top Patients by Encounter Count")
    
    # Get top patients
    top_patients = repeated_counts.head(n_top_patients).index.tolist()
    df_top = df[df['patient_nbr'].isin(top_patients)].copy()
    
    # Create interactive bar chart
    top_patients_df = pd.DataFrame({
        'patient_nbr': repeated_counts.head(n_top_patients).index,
        'encounter_count': repeated_counts.head(n_top_patients).values
    })
    
    fig_patients = px.bar(
        top_patients_df,
        x='patient_nbr',
        y='encounter_count',
        title=f"Top {n_top_patients} Patients by Encounter Count",
        labels={'encounter_count': 'Number of Encounters', 'patient_nbr': 'Patient ID'},
        color='encounter_count',
        color_continuous_scale='Reds'
    )
    
    fig_patients.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_patients, use_container_width=True)

    # Patient details expander
    with st.expander("📋 Patient Encounter Details"):
        selected_patient = st.selectbox(
            "Select a patient to view details:",
            top_patients_df['patient_nbr'].tolist()
        )
        
        if selected_patient:
            patient_data = df[df['patient_nbr'] == selected_patient].sort_values('encounter_id')
            
            # Show readmission pattern
            readmit_pattern = patient_data['readmitted'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Encounters", len(patient_data))
            with col2:
                readmit_rate = (patient_data['readmitted'] != 'NO').mean() * 100
                st.metric("Readmission Rate", f"{readmit_rate:.1f}%")
            
            # Show readmission timeline
            st.markdown("**Readmission Pattern:**")
            
            # Create a simple timeline visualization
            timeline_data = []
            for idx, (_, row) in enumerate(patient_data.iterrows()):
                timeline_data.append({
                    'Encounter': idx + 1,
                    'Status': row['readmitted'],
                    'Color': READMISSION_COLORS[row['readmitted']]
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            
            fig_timeline = go.Figure()
            
            for status in timeline_df['Status'].unique():
                mask = timeline_df['Status'] == status
                fig_timeline.add_trace(go.Scatter(
                    x=timeline_df[mask]['Encounter'],
                    y=[1] * sum(mask),
                    mode='markers',
                    marker=dict(size=20, color=READMISSION_COLORS[status]),
                    name=status,
                    hovertemplate='Encounter %{x}<br>Status: ' + status + '<extra></extra>'
                ))
            
            fig_timeline.update_layout(
                height=150,
                showlegend=True,
                yaxis=dict(visible=False, range=[0.5, 1.5]),
                xaxis=dict(title='Encounter Number'),
                margin=dict(l=0, r=0, t=20, b=40)
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)

# Additional analysis section
st.markdown("---")
st.subheader("🔍 Key Insights for Repeated Patients")

# Create three columns for insights
insight_col1, insight_col2, insight_col3 = st.columns(3)

with insight_col1:
    # Readmission breakdown
    readmit_breakdown = df_repeated['readmitted'].value_counts()
    
    # Create a proper dataframe for the pie chart
    pie_data = pd.DataFrame({
        'Status': readmit_breakdown.index,
        'Count': readmit_breakdown.values
    })
    
    fig_pie = px.pie(
        pie_data,
        values='Count',
        names='Status',
        title="Readmission Status Distribution",
        color='Status',  # This is important!
        color_discrete_map=READMISSION_COLORS,
        hole=0.4
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(height=300)
    st.plotly_chart(fig_pie, use_container_width=True)

with insight_col2:
    # Average time in hospital by readmission status
    avg_time = df_repeated.groupby('readmitted')['time_in_hospital'].mean().round(1)
    
    fig_avg_time = px.bar(
        x=avg_time.index,
        y=avg_time.values,
        title="Avg Time in Hospital by Readmission",
        labels={'x': 'Readmission Status', 'y': 'Days'},
        color=avg_time.index,
        color_discrete_map=READMISSION_COLORS
    )
    fig_avg_time.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_avg_time, use_container_width=True)

with insight_col3:
    # Number of procedures by readmission status
    avg_procedures = df_repeated.groupby('readmitted')['num_procedures'].mean().round(1)
    
    fig_procedures = px.bar(
        x=avg_procedures.index,
        y=avg_procedures.values,
        title="Avg Procedures by Readmission",
        labels={'x': 'Readmission Status', 'y': 'Number of Procedures'},
        color=avg_procedures.index,
        color_discrete_map=READMISSION_COLORS
    )
    fig_procedures.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_procedures, use_container_width=True)

# Feature correlation analysis
st.markdown("---")
st.subheader("🔗 Key Feature Relationships")

# Get numeric features
numeric_features = [col for col in df_repeated.columns if pd.api.types.is_numeric_dtype(df_repeated[col])]
numeric_features = [f for f in numeric_features if f not in ['encounter_id', 'patient_nbr', 
                                                               'discharge_disposition_id', 
                                                               'admission_type_id', 
                                                               'admission_source_id']]

if len(numeric_features) >= 2:
    # Calculate correlations with readmission (encoded)
    df_temp = df_repeated.copy()
    df_temp['readmitted_encoded'] = df_temp['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})
    
    # Get correlations with readmission
    readmit_correlations = df_temp[numeric_features].corrwith(df_temp['readmitted_encoded']).abs().sort_values(ascending=False)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Features Most Associated with Readmission")
        
        # Top 5 features correlated with readmission
        top_features = readmit_correlations.head(5)
        
        # Create a horizontal bar chart
        fig_corr = go.Figure(data=[
            go.Bar(
                x=top_features.values,
                y=top_features.index,
                orientation='h',
                marker_color='lightcoral',
                text=[f'{v:.3f}' for v in top_features.values],
                textposition='auto',
            )
        ])
        
        fig_corr.update_layout(
            title="Top Features Correlated with Readmission Risk",
            xaxis_title="Correlation Strength",
            yaxis_title="",
            height=300,
            showlegend=False,
            xaxis=dict(range=[0, max(top_features.values) * 1.1])
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        st.markdown("### 🔍 Feature Relationship Insights")
        
        # Calculate some key relationships
        insights = []
        
        # Time in hospital vs procedures
        if 'time_in_hospital' in numeric_features and 'num_procedures' in numeric_features:
            corr = df_repeated[['time_in_hospital', 'num_procedures']].corr().iloc[0, 1]
            insights.append(f"**Hospital Stay vs Procedures**: {corr:.2f} correlation")
            if corr > 0.5:
                insights.append("→ Longer stays associated with more procedures")
        
        # Number of medications
        med_count = df_repeated['number_diagnoses'].mean() if 'number_diagnoses' in df_repeated.columns else None
        if med_count:
            insights.append(f"**Avg Diagnoses**: {med_count:.1f} per encounter")
        
        # Lab procedures
        if 'num_lab_procedures' in df_repeated.columns:
            high_lab = df_repeated[df_repeated['readmitted'] == '<30']['num_lab_procedures'].mean()
            low_lab = df_repeated[df_repeated['readmitted'] == 'NO']['num_lab_procedures'].mean()
            if high_lab and low_lab:
                diff = ((high_lab - low_lab) / low_lab * 100)
                insights.append(f"**Lab Procedures**: {diff:+.0f}% for <30 day readmits")
        
        # Create insight cards
        for insight in insights[:4]:  # Show top 4 insights
            st.info(insight)
        
        # Add actionable insight
        st.success("💡 **Key Finding**: Focus on patients with " + 
                  f"{top_features.index[0].replace('_', ' ')} " +
                  "for early intervention")

# Quick correlation matrix for selected features only
with st.expander("🔧 Explore Detailed Correlations", expanded=False):
    st.markdown("### Select Features for Correlation Matrix")
    
    selected_corr_features = st.multiselect(
        "Choose features to analyze relationships:",
        numeric_features,
        default=numeric_features[:6] if len(numeric_features) >= 6 else numeric_features,
        help="Select 2-10 features to see their correlations"
    )
    
    if len(selected_corr_features) >= 2:
        # Calculate correlation matrix
        corr_matrix = df_repeated[selected_corr_features].corr()
        
        # Create a simpler heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation", len=0.5)
        ))
        
        fig_heatmap.update_layout(
            title="Feature Correlation Matrix",
            height=400,
            width=500,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Show strongest correlations
        st.markdown("**Strongest Correlations:**")
        
        # Get correlation pairs
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)
        
        # Show top 3
        for _, row in corr_df.head(3).iterrows():
            if abs(row['Correlation']) > 0.3:
                direction = "↑↑" if row['Correlation'] > 0 else "↑↓"
                st.write(f"• **{row['Feature 1']}** {direction} **{row['Feature 2']}**: {row['Correlation']:.2f}")

# Footer with summary statistics
st.markdown("---")
with st.expander("📊 Dataset Summary Statistics", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Repeated Patients Dataset:**")
        st.write(f"- Total unique patients: {df_repeated['patient_nbr'].nunique():,}")
        st.write(f"- Total encounters: {len(df_repeated):,}")
        st.write(f"- Features: {len(df_repeated.columns)}")
        st.write(f"- Date range: Check admission dates if available")
    
    with col2:
        st.markdown("**Readmission Statistics:**")
        readmit_stats = df_repeated['readmitted'].value_counts(normalize=True) * 100
        for status, pct in readmit_stats.items():
            status_display = status.replace(">", "\\>") if status.startswith(">") else status
            st.write(f"- {status_display}: {pct:.1f}%")

# Note about additional features
st.info("💡 **Note**: This dashboard focuses on repeated patients (those with multiple encounters). Additional analyses and features can be added to explore single-encounter patients, temporal patterns, and predictive insights.")

# Diagnosis Pattern Analysis
st.markdown("---")
st.header("🏥 Diagnosis Pattern Analysis")

# Define the truncate_text function (for use in tables/specific areas)
def truncate_text(text, max_length=20):
    """Truncate text to specified length and add ellipsis if needed."""
    if len(text) > max_length:
        return text[:max_length-3] + "..."
    return text

if 'diag_1_category' in df_repeated.columns:
    # Create tabs for different views
    diag_tab1, diag_tab2, diag_tab3 = st.tabs(["Overview", "Detailed Analysis", "Risk Rankings"])
    
    with diag_tab1:
        # Count by diagnosis category and readmission
        diag_readmit = df_repeated.groupby(['diag_1_category', 'readmitted']).size().reset_index(name='count')
        
        # Create stacked bar chart with proper colors
        fig_diag = go.Figure()
        
        # Add traces for each readmission status in specific order
        for status in ['NO', '>30', '<30']:  # Stack order: NO at bottom, <30 at top
            if status in diag_readmit['readmitted'].unique():
                data_subset = diag_readmit[diag_readmit['readmitted'] == status]
                fig_diag.add_trace(go.Bar(
                    x=data_subset['diag_1_category'],
                    y=data_subset['count'],
                    name=status,
                    marker_color=READMISSION_COLORS[status],
                    text=data_subset['count'],
                    textposition='inside',
                    textfont=dict(size=10)
                ))
        
        fig_diag.update_layout(
            title='Primary Diagnosis Distribution by Readmission Status',
            xaxis_title='Diagnosis Category',
            yaxis_title='Number of Encounters',
            barmode='stack',
            height=500,
            xaxis_tickangle=-45,
            showlegend=True
        )
        
        st.plotly_chart(fig_diag, use_container_width=True)
        
        # Better Quick Insights using HTML/CSS for full text display
        st.subheader("📊 Quick Insights")
        
        # Calculate key metrics
        top_diagnosis = df_repeated['diag_1_category'].value_counts().index[0]
        top_diag_count = df_repeated['diag_1_category'].value_counts().iloc[0]
        
        readmit_by_diag = df_repeated.groupby('diag_1_category').agg({
            'readmitted': lambda x: (x != 'NO').mean() * 100,
            'patient_nbr': 'count'
        })
        readmit_by_diag = readmit_by_diag[readmit_by_diag['patient_nbr'] >= 50]
        highest_risk_diag = readmit_by_diag['readmitted'].idxmax()
        highest_risk_rate = readmit_by_diag['readmitted'].max()
        unique_diagnoses = df_repeated['diag_1_category'].nunique()
        avg_readmit = (df_repeated['readmitted'] != 'NO').mean() * 100
        
        # Custom CSS for better display
        st.markdown("""
        <style>
            .insight-card {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .insight-title {
                font-size: 14px;
                color: #666;
                margin-bottom: 8px;
                font-weight: 500;
            }
            .insight-value {
                font-size: 18px;
                font-weight: bold;
                color: #333;
                margin-bottom: 5px;
                word-wrap: break-word;
                line-height: 1.3;
            }
            .insight-detail {
                font-size: 13px;
                color: #28a745;
                margin-top: 5px;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Create two rows for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">Most Common Diagnosis</div>
                <div class="insight-value">{top_diagnosis}</div>
                <div class="insight-detail">↑ {top_diag_count:,} cases ({(top_diag_count/len(df_repeated)*100):.1f}% of all)</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">Total Diagnosis Categories</div>
                <div class="insight-value">{unique_diagnoses} Categories</div>
                <div class="insight-detail">↑ Across all patient encounters</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">Highest Risk Diagnosis</div>
                <div class="insight-value">{highest_risk_diag}</div>
                <div class="insight-detail">⚠️ {highest_risk_rate:.1f}% readmission rate</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">Overall Readmission Rate</div>
                <div class="insight-value">{avg_readmit:.1f}%</div>
                <div class="insight-detail">↑ Across all diagnoses</div>
            </div>
            """, unsafe_allow_html=True)
    
    with diag_tab2:
        # Interactive diagnosis explorer
        st.subheader("🔍 Explore Specific Diagnosis")
        
        # Dropdown to select diagnosis
        selected_diagnosis = st.selectbox(
            "Select a diagnosis category to analyze:",
            sorted(df_repeated['diag_1_category'].unique())
        )
        
        # Filter data for selected diagnosis
        diag_data = df_repeated[df_repeated['diag_1_category'] == selected_diagnosis]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Readmission breakdown for this diagnosis - with proper colors
            readmit_counts = diag_data['readmitted'].value_counts()
            
            # Create pie chart with proper colors
            pie_data = pd.DataFrame({
                'Status': readmit_counts.index,
                'Count': readmit_counts.values
            })
            
            fig_pie = px.pie(
                pie_data,
                values='Count',
                names='Status',
                title=f"Readmission Breakdown: {selected_diagnosis}",
                color='Status',
                color_discrete_map=READMISSION_COLORS,
                hole=0.4
            )
            
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Stats for this diagnosis with better formatting
            st.markdown(f"### 📈 Statistics")
            
            total_encounters = len(diag_data)
            readmit_rate = (diag_data['readmitted'] != 'NO').mean() * 100
            urgent_rate = (diag_data['readmitted'] == '<30').mean() * 100
            avg_los = diag_data['time_in_hospital'].mean()
            avg_procedures = diag_data['num_procedures'].mean()
            
            # Display selected diagnosis name (wrapped if long)
            st.markdown(f"**{selected_diagnosis}**")
            st.markdown("---")
            
            # Use columns for cleaner layout
            st.metric("Total Cases", f"{total_encounters:,}")
            
            # Color-coded readmission metrics
            if readmit_rate > 80:
                st.metric("Readmission", f"{readmit_rate:.1f}%", "⚠️ High")
            elif readmit_rate > 60:
                st.metric("Readmission", f"{readmit_rate:.1f}%", "⚡ Moderate")
            else:
                st.metric("Readmission", f"{readmit_rate:.1f}%", "✓ Lower")
            
            st.metric("Urgent (<30)", f"{urgent_rate:.1f}%")
            st.metric("Avg Stay", f"{avg_los:.1f} days")
            st.metric("Avg Procedures", f"{avg_procedures:.1f}")
    
    with diag_tab3:
        # Risk rankings
        st.subheader("⚠️ Diagnosis Risk Rankings")
        
        # Calculate comprehensive risk scores
        risk_analysis = df_repeated.groupby('diag_1_category').agg({
            'readmitted': [
                lambda x: (x != 'NO').mean() * 100,  # Overall readmission
                lambda x: (x == '<30').mean() * 100,  # Urgent readmission
                'count'  # Total encounters
            ],
            'time_in_hospital': 'mean',
            'num_procedures': 'mean'
        }).round(1)
        
        # Flatten column names
        risk_analysis.columns = ['Readmit_Rate', 'Urgent_Rate', 'Total_Count', 'Avg_LOS', 'Avg_Procedures']
        
        # Filter for statistical significance
        risk_analysis = risk_analysis[risk_analysis['Total_Count'] >= 50]
        
        # Create risk score (weighted combination)
        risk_analysis['Risk_Score'] = (
            risk_analysis['Readmit_Rate'] * 0.4 + 
            risk_analysis['Urgent_Rate'] * 0.6
        ).round(1)
        
        # Sort by risk score
        risk_analysis = risk_analysis.sort_values('Risk_Score', ascending=False)
        
        # Create visual risk chart
        fig_risk = px.scatter(
            risk_analysis.reset_index(),
            x='Urgent_Rate',
            y='Readmit_Rate',
            size='Total_Count',
            color='Risk_Score',
            hover_data=['diag_1_category', 'Total_Count', 'Avg_LOS'],
            title='Diagnosis Risk Matrix (size = number of cases)',
            labels={
                'Urgent_Rate': 'Urgent Readmission Rate (<30 days) %',
                'Readmit_Rate': 'Overall Readmission Rate %',
                'Risk_Score': 'Risk Score',
                'diag_1_category': 'Diagnosis'
            },
            color_continuous_scale='Reds',
            size_max=50
        )
        
        # Add quadrant lines
        fig_risk.add_hline(y=risk_analysis['Readmit_Rate'].median(), line_dash="dash", line_color="gray", opacity=0.5)
        fig_risk.add_vline(x=risk_analysis['Urgent_Rate'].median(), line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add annotations for quadrants
        fig_risk.add_annotation(x=5, y=85, text="High Risk", showarrow=False, font=dict(size=12, color="red"))
        fig_risk.add_annotation(x=35, y=85, text="Critical Risk", showarrow=False, font=dict(size=12, color="darkred"))
        
        fig_risk.update_layout(height=500)
        
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Top risk diagnoses table
        st.markdown("### 🎯 Focus Areas - Top 10 Highest Risk Diagnoses")
        
        display_df = risk_analysis.head(10)[['Risk_Score', 'Readmit_Rate', 'Urgent_Rate', 'Total_Count', 'Avg_LOS']]
        display_df.columns = ['Risk Score', 'Readmit %', '<30 Day %', 'Cases', 'Avg Days']
        
        # Reset index to show diagnosis names
        display_df = display_df.reset_index()
        display_df.columns = ['Diagnosis'] + list(display_df.columns[1:])
        
        # Style the dataframe
        st.dataframe(
            display_df.style.format({
                'Risk Score': '{:.1f}',
                'Readmit %': '{:.1f}%',
                '<30 Day %': '{:.1f}%',
                'Cases': '{:,.0f}',
                'Avg Days': '{:.1f}'
            }).background_gradient(cmap='Reds', subset=['Risk Score']),
            use_container_width=True,
            height=400
        )
        
        # Actionable insight
        top_risk_diag = risk_analysis.index[0]
        top_risk_score = risk_analysis.iloc[0]['Risk_Score']
        top_urgent_rate = risk_analysis.iloc[0]['Urgent_Rate']
        
        st.success(f"""
        💡 **Recommendation**: Focus intervention efforts on **{top_risk_diag}** patients.
        
        - **Risk Score**: {top_risk_score:.1f}/100
        - **Urgent Readmission**: {top_urgent_rate:.1f}% within 30 days
        - **Action**: Implement specialized discharge protocols and follow-up care
        """)
        
        # Additional insights
        with st.expander("📋 Additional Risk Insights"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔴 High Risk Categories (>40% readmission)**")
                high_risk = risk_analysis[risk_analysis['Readmit_Rate'] > 40].index.tolist()
                if high_risk:
                    for diag in high_risk[:5]:
                        rate = risk_analysis.loc[diag, 'Readmit_Rate']
                        st.write(f"- {diag}: {rate:.1f}%")
                else:
                    st.write("No diagnoses exceed 40% readmission rate")
            
            with col2:
                st.markdown("**⚡ Urgent Risk Categories (>20% <30 day)**")
                urgent_risk = risk_analysis[risk_analysis['Urgent_Rate'] > 20].index.tolist()
                if urgent_risk:
                    for diag in urgent_risk[:5]:
                        rate = risk_analysis.loc[diag, 'Urgent_Rate']
                        st.write(f"- {diag}: {rate:.1f}%")
                else:
                    st.write("No diagnoses exceed 20% urgent readmission rate")

else:
    st.info("Diagnosis category information not available in the dataset.")

# Medication Analysis
st.markdown("---")
st.header("💊 Medication Impact Analysis")

# Get medication columns - look for common patterns
# First, let's see what columns might be medications
potential_med_columns = []

# Common medication column patterns
medication_keywords = ['insulin', 'metformin', 'glipizide', 'glyburide', 'pioglitazone', 
                      'rosiglitazone', 'medication', 'drug', 'med']

# Find columns that might be medications
for col in df_repeated.columns:
    col_lower = col.lower()
    if any(keyword in col_lower for keyword in medication_keywords):
        potential_med_columns.append(col)

# Also check for columns with Yes/No/Steady/Up/Down values (common for medications)
for col in df_repeated.columns:
    if df_repeated[col].dtype == 'object':
        unique_vals = df_repeated[col].dropna().unique()
        if len(unique_vals) <= 5 and any(val in ['Yes', 'No', 'Steady', 'Up', 'Down'] for val in unique_vals):
            if col not in potential_med_columns and col != 'readmitted':
                potential_med_columns.append(col)

if potential_med_columns:
    # Let user select which columns are medications
    med_columns = st.multiselect(
        "Select medication columns to analyze:",
        potential_med_columns,
        default=potential_med_columns[:6]  # Select first 6 by default
    )
    
    if med_columns:
        med_tab1, med_tab2 = st.tabs(["Medication Usage Overview", "Medication Combinations"])
        
        with med_tab1:
            # Medication usage by readmission status
            med_data = []
            for med in med_columns:
                # Get value counts for this medication
                med_values = df_repeated[med].value_counts()
                
                for status in df_repeated['readmitted'].unique():
                    subset = df_repeated[df_repeated['readmitted'] == status][med]
                    
                    # Handle different types of medication data
                    if 'Yes' in subset.unique() or 'yes' in subset.unique():
                        # Binary Yes/No
                        yes_count = ((subset == 'Yes') | (subset == 'yes')).sum()
                        total_count = subset.notna().sum()
                        usage_rate = (yes_count / total_count * 100) if total_count > 0 else 0
                        
                        med_data.append({
                            'Medication': med.replace('_', ' ').title(),
                            'Readmission Status': status,
                            'Usage Rate': usage_rate
                        })
                    elif any(val in ['Up', 'Down', 'Steady'] for val in subset.unique()):
                        # Dosage changes
                        for change_type in ['Up', 'Down', 'Steady']:
                            if change_type in subset.unique():
                                change_count = (subset == change_type).sum()
                                total_count = subset.notna().sum()
                                change_rate = (change_count / total_count * 100) if total_count > 0 else 0
                                
                                med_data.append({
                                    'Medication': f"{med.replace('_', ' ').title()} - {change_type}",
                                    'Readmission Status': status,
                                    'Usage Rate': change_rate
                                })
            
            if med_data:
                med_df = pd.DataFrame(med_data)
                
                fig_med = px.bar(
                    med_df,
                    x='Medication',
                    y='Usage Rate',
                    color='Readmission Status',
                    barmode='group',
                    title='Medication Usage Rates by Readmission Status',
                    color_discrete_map=READMISSION_COLORS,
                    labels={'Usage Rate': 'Usage Rate (%)'}
                )
                
                fig_med.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_med, use_container_width=True)
            else:
                st.warning("No medication usage data found in the selected columns.")
        
        with med_tab2:
            # Analyze medication combinations
            st.markdown("**Common Medication Combinations:**")
            
            # Create binary columns for medications - PRESERVE INDEX
            med_binary = pd.DataFrame(index=df_repeated.index)  # Important: preserve the original index
            
            for med in med_columns:
                if 'Yes' in df_repeated[med].unique() or 'yes' in df_repeated[med].unique():
                    med_binary[med] = ((df_repeated[med] == 'Yes') | (df_repeated[med] == 'yes')).astype(int)
                else:
                    # For other types, just check if not null/No
                    med_binary[med] = (~df_repeated[med].isin(['No', 'no', None, np.nan])).astype(int)
            
            if not med_binary.empty:
                # Find common combinations
                med_combinations = med_binary.value_counts().head(10)
                
                # Create a more readable format
                combination_data = []
                for combo, count in med_combinations.items():
                    if isinstance(combo, tuple):
                        meds_used = [med_columns[i].replace('_', ' ').title() 
                                for i, val in enumerate(combo) if val == 1]
                    else:
                        meds_used = [med_columns[0].replace('_', ' ').title()] if combo == 1 else []
                    
                    if meds_used:  # Only show combinations with at least one medication
                        # Get readmission rate for this combination - FIXED VERSION
                        mask = pd.Series([True] * len(df_repeated), index=df_repeated.index)  # Preserve index
                        
                        for i, med in enumerate(med_columns):
                            if isinstance(combo, tuple):
                                mask = mask & (med_binary[med] == combo[i])
                            else:
                                mask = mask & (med_binary[med] == combo)
                        
                        # Now mask has the same index as df_repeated
                        readmit_rate = (df_repeated.loc[mask, 'readmitted'] != 'NO').mean() * 100
                        
                        combination_data.append({
                            'Medications': ', '.join(meds_used),
                            'Count': count,
                            'Percentage': (count / len(df_repeated)) * 100,
                            'Readmission Rate': readmit_rate
                        })
                
                if combination_data:
                    combo_df = pd.DataFrame(combination_data)
                    st.dataframe(
                        combo_df.style.format({
                            'Percentage': '{:.1f}%',
                            'Readmission Rate': '{:.1f}%'
                        }).background_gradient(cmap='YlOrRd', subset=['Readmission Rate']),
                        use_container_width=True
                    )
                else:
                    st.info("No medication combinations found.")
            else:
                st.info("Unable to analyze medication combinations with the selected columns.")
    else:
        st.info("Please select medication columns from the dropdown above to see the analysis.")
else:
    st.warning("""
    📊 **No medication columns detected in the dataset.**
    
    The analysis looks for columns containing medication names like 'insulin', 'metformin', etc., 
    or columns with values like 'Yes', 'No', 'Up', 'Down', 'Steady'.
    
    If your dataset has medication information with different column names, 
    please check the debug expander above to see available columns.
    """)
# Age Group Analysis
st.markdown("---")
st.header("👥 Age Group Analysis")

if 'age' in df_repeated.columns:
    # Check if age is already in groups or numeric
    df_repeated_copy = df_repeated.copy()
    
    # If age is already in string format (age ranges), use it directly
    if df_repeated_copy['age'].dtype == 'object':
        # Age is already in groups like '[0-10)', '[10-20)', etc.
        df_repeated_copy['age_group'] = df_repeated_copy['age']
    else:
        # If age is numeric, create age groups
        age_bins = [0, 30, 40, 50, 60, 70, 80, 100]
        age_labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
        df_repeated_copy['age_group'] = pd.cut(df_repeated_copy['age'], bins=age_bins, labels=age_labels)
    
    age_col1, age_col2 = st.columns([3, 2])
    
    with age_col1:
        # Age distribution by readmission
        age_readmit = df_repeated_copy.groupby(['age_group', 'readmitted']).size().reset_index(name='count')
        
        fig_age = px.bar(
            age_readmit,
            x='age_group',
            y='count',
            color='readmitted',
            title='Age Distribution by Readmission Status',
            color_discrete_map=READMISSION_COLORS,
            labels={'age_group': 'Age Group', 'count': 'Number of Encounters'},
            barmode='group'
        )
        
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)

# Risk Factor Summary
st.markdown("---")
st.header("⚠️ Risk Factor Summary")

risk_col1, risk_col2, risk_col3 = st.columns(3)

with risk_col1:
    st.markdown("### 🔴 High Risk Factors")
    st.markdown("""
    Factors associated with <30 day readmission:
    - Multiple prior encounters
    - Longer initial hospital stay
    - Higher number of procedures
    - Certain diagnosis categories
    - Complex medication regimens
    """)

with risk_col2:
    st.markdown("### 🟡 Moderate Risk Factors")
    st.markdown("""
    Factors associated with >30 day readmission:
    - Previous readmissions
    - Specific age groups
    - Certain medication changes
    - Comorbidity patterns
    """)

with risk_col3:
    st.markdown("### 🟢 Protective Factors")
    st.markdown("""
    Factors associated with no readmission:
    - Appropriate medication management
    - Shorter hospital stays
    - Fewer procedures
    - Certain admission types
    """)

# Export Options
st.markdown("---")
st.header("📥 Export Options")

export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    # Export filtered data
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        label="📊 Download Filtered Data (CSV)",
        data=csv,
        file_name='repeated_patients_filtered.csv',
        mime='text/csv'
    )

with export_col2:
    # Export summary statistics
    summary_stats = {
        'Total Repeated Patients': len(repeated_counts[repeated_counts > 1]),
        'Total Encounters': df_repeated.shape[0],
        'Readmission Rate': readmit_rate,
        'Avg Encounters per Patient': avg_encounters
    }
    summary_df = pd.DataFrame(summary_stats.items(), columns=['Metric', 'Value'])
    summary_csv = summary_df.to_csv(index=False)
    
    st.download_button(
        label="📈 Download Summary Stats (CSV)",
        data=summary_csv,
        file_name='repeated_patients_summary.csv',
        mime='text/csv'
    )

with export_col3:
    # Generate report placeholder
    if st.button("📄 Generate Full Report", type="primary"):
        st.info("Report generation feature coming soon!")

# Footer
st.markdown("---")
st.caption("Repeated Patient Analysis Dashboard | Data updated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

# Add custom CSS for better styling
st.markdown("""
<style>
    /* Custom styling for metrics */
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Style for tab containers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)
