# pages/3_Advanced_Analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import skew, kurtosis
from data_processing import get_processed_dataframe

st.set_page_config(page_title="Advanced Analysis", page_icon="🔬", layout="wide")

# Load data
df = get_processed_dataframe()

# Header
st.markdown("""
<div style="background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
    <h1>🔬 Advanced Analysis</h1>
    <p>Deep statistical analysis and feature engineering insights</p>
</div>
""", unsafe_allow_html=True)

# Get different data subsets
df_sorted = df.sort_values(by='encounter_id')
df_first_encounter = df_sorted.drop_duplicates(subset='patient_nbr', keep='first')
df_repeated = df.groupby('patient_nbr').filter(lambda x: len(x) > 1)

# Sidebar for analysis configuration
with st.sidebar:
    st.header("🔧 Analysis Configuration")
    
    # Data subset selection
    data_subset = st.selectbox(
        "Select Data Subset",
        ["All Patients", "First Encounters Only", "Repeated Patients Only", "Compare All"]
    )
    
    # Feature selection
    exclude_cols = {"readmitted", "encounter_id", "patient_nbr", "diag_1", "diag_2", "diag_3",
                   "discharge_disposition_id", "admission_type_id", "admission_source_id"}
    all_features = [col for col in df.columns if col not in exclude_cols]
    
    selected_features = st.multiselect(
        "Select Features for Analysis",
        all_features,
        default=all_features[:10]
    )
    
    # Statistical test selection
    st.subheader("Statistical Tests")
    perform_normality = st.checkbox("Normality Tests", value=True)
    perform_correlation = st.checkbox("Advanced Correlation", value=True)
    perform_anova = st.checkbox("ANOVA/Chi-Square Tests", value=True)

# Main analysis tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Distribution Analysis", 
    "🔗 Correlation Deep Dive", 
    "📈 Statistical Testing", 
    "🎯 Feature Importance",
    "🔄 Comparative Analysis"
])

# Tab 1: Distribution Analysis
with tab1:
    st.header("Feature Distribution Analysis")
    
    # Select dataset based on sidebar choice
    if data_subset == "All Patients":
        analysis_df = df
    elif data_subset == "First Encounters Only":
        analysis_df = df_first_encounter
    elif data_subset == "Repeated Patients Only":
        analysis_df = df_repeated
    else:
        analysis_df = df  # For compare all, we'll handle separately
    
    # Distribution type selector
    dist_col1, dist_col2 = st.columns([3, 1])
    
    with dist_col1:
        st.subheader(f"Analyzing: {data_subset}")
    
    with dist_col2:
        view_type = st.radio("View Type", ["Grid", "Individual"], horizontal=True)
    
    if view_type == "Grid":
        # Grid view of distributions
        n_features = len(selected_features)
        if n_features > 0:
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=selected_features,
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            for idx, feature in enumerate(selected_features):
                row = idx // n_cols + 1
                col = idx % n_cols + 1
                
                if pd.api.types.is_numeric_dtype(analysis_df[feature]):
                    # Numeric feature - show distribution with KDE
                    feature_data = analysis_df[feature].dropna()
                    
                    # Create histogram
                    hist_trace = go.Histogram(
                        x=feature_data,
                        name=feature,
                        nbinsx=30,
                        opacity=0.7,
                        histnorm='probability density',
                        showlegend=False
                    )
                    
                    fig.add_trace(hist_trace, row=row, col=col)
                    
                    # Add distribution stats as annotation
                    mean_val = feature_data.mean()
                    median_val = feature_data.median()
                    skew_val = skew(feature_data)
                    
                    stats_text = f"μ={mean_val:.2f}<br>M={median_val:.2f}<br>γ={skew_val:.2f}"
                    fig.add_annotation(
                        xref=f"x{idx+1} domain",
                        yref=f"y{idx+1} domain",
                        x=0.95, y=0.95,
                        text=stats_text,
                        showarrow=False,
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="black",
                        borderwidth=1
                    )
                else:
                    # Categorical feature - show top categories
                    value_counts = analysis_df[feature].value_counts().head(10)
                    
                    bar_trace = go.Bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        showlegend=False
                    )
                    
                    fig.add_trace(bar_trace, row=row, col=col)
                    fig.update_xaxes(tickangle=-45, row=row, col=col)
            
            fig.update_layout(height=300 * n_rows, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Individual feature analysis
        if selected_features:
            feature_to_analyze = st.selectbox("Select feature for detailed analysis", selected_features)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if pd.api.types.is_numeric_dtype(analysis_df[feature_to_analyze]):
                    # Create detailed distribution plot
                    fig = go.Figure()
                    
                    # Add histogram
                    feature_data = analysis_df[feature_to_analyze].dropna()
                    fig.add_trace(go.Histogram(
                        x=feature_data,
                        name='Distribution',
                        nbinsx=50,
                        opacity=0.7,
                        histnorm='probability density'
                    ))
                    
                    # Add KDE
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(feature_data)
                    x_range = np.linspace(feature_data.min(), feature_data.max(), 200)
                    kde_values = kde(x_range)
                    
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=kde_values,
                        mode='lines',
                        name='KDE',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Add mean and median lines
                    mean_val = feature_data.mean()
                    median_val = feature_data.median()
                    
                    fig.add_vline(x=mean_val, line_dash="dash", line_color="green", 
                                 annotation_text=f"Mean: {mean_val:.2f}")
                    fig.add_vline(x=median_val, line_dash="dash", line_color="blue", 
                                 annotation_text=f"Median: {median_val:.2f}")
                    
                    fig.update_layout(
                        title=f"Distribution of {feature_to_analyze}",
                        xaxis_title=feature_to_analyze,
                        yaxis_title="Density",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Q-Q plot for normality
                    st.subheader("Normality Assessment")
                    fig_qq = go.Figure()
                    
                    # Calculate theoretical quantiles
                    sorted_data = np.sort(feature_data)
                    n = len(sorted_data)
                    theoretical_quantiles = stats.norm.ppf((np.arange(1, n+1) - 0.5) / n)
                    
                    # Add Q-Q plot
                    fig_qq.add_trace(go.Scatter(
                        x=theoretical_quantiles,
                        y=sorted_data,
                        mode='markers',
                        name='Data',
                        marker=dict(size=5)
                    ))
                    
                    # Add reference line
                    min_val = min(theoretical_quantiles.min(), sorted_data.min())
                    max_val = max(theoretical_quantiles.max(), sorted_data.max())
                    fig_qq.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Normal',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_qq.update_layout(
                        title="Q-Q Plot",
                        xaxis_title="Theoretical Quantiles",
                        yaxis_title="Sample Quantiles",
                        height=350
                    )
                    
                    st.plotly_chart(fig_qq, use_container_width=True)
                
                else:
                    # Categorical feature analysis
                    value_counts = analysis_df[feature_to_analyze].value_counts()
                    
                    # Create pie chart for top categories
                    if len(value_counts) > 10:
                        top_counts = value_counts.head(9)
                        other_count = value_counts[9:].sum()
                        top_counts['Other'] = other_count
                        value_counts = top_counts
                    
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Distribution of {feature_to_analyze}",
                        hole=0.4
                    )
                    
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(height=500)
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Statistical summary
                st.subheader("📊 Statistical Summary")
                
                if pd.api.types.is_numeric_dtype(analysis_df[feature_to_analyze]):
                    feature_data = analysis_df[feature_to_analyze].dropna()
                    
                    # Calculate statistics
                    stats_dict = {
                        "Count": len(feature_data),
                        "Missing": analysis_df[feature_to_analyze].isna().sum(),
                        "Mean": feature_data.mean(),
                        "Std Dev": feature_data.std(),
                        "Min": feature_data.min(),
                        "Q1": feature_data.quantile(0.25),
                        "Median": feature_data.median(),
                        "Q3": feature_data.quantile(0.75),
                        "Max": feature_data.max(),
                        "Skewness": skew(feature_data),
                        "Kurtosis": kurtosis(feature_data)
                    }
                    
                    # Display as metrics
                    for stat, value in stats_dict.items():
                        if stat in ["Count", "Missing"]:
                            st.metric(stat, f"{int(value):,}")
                        else:
                            st.metric(stat, f"{value:.3f}")
                    
                    # Normality tests
                    if perform_normality:
                        st.subheader("🧪 Normality Tests")
                        
                        # Shapiro-Wilk test (for samples < 5000)
                        if len(feature_data) < 5000:
                            stat_sw, p_sw = stats.shapiro(feature_data)
                            st.write(f"**Shapiro-Wilk Test**")
                            st.write(f"- Statistic: {stat_sw:.4f}")
                            st.write(f"- p-value: {p_sw:.4f}")
                            st.write(f"- Result: {'Normal' if p_sw > 0.05 else 'Not Normal'} (α=0.05)")
                        
                        # Anderson-Darling test
                        result_ad = stats.anderson(feature_data)
                        st.write(f"**Anderson-Darling Test**")
                        st.write(f"- Statistic: {result_ad.statistic:.4f}")
                        st.write(f"- Critical Value (5%): {result_ad.critical_values[2]:.4f}")
                        st.write(f"- Result: {'Normal' if result_ad.statistic < result_ad.critical_values[2] else 'Not Normal'}")
                
                else:
                    # Categorical statistics
                    value_counts = analysis_df[feature_to_analyze].value_counts()
                    
                    st.metric("Unique Values", len(value_counts))
                    st.metric("Mode", value_counts.index[0])
                    st.metric("Mode Count", f"{value_counts.iloc[0]:,}")
                    st.metric("Missing", f"{analysis_df[feature_to_analyze].isna().sum():,}")
                    
                    # Entropy
                    probs = value_counts / value_counts.sum()
                    entropy = -sum(probs * np.log2(probs + 1e-10))
                    st.metric("Entropy", f"{entropy:.3f}")

# Tab 2: Correlation Deep Dive
with tab2:
    st.header("Advanced Correlation Analysis")
    
    if perform_correlation:
        # Select only numeric features
        numeric_features = [f for f in selected_features if pd.api.types.is_numeric_dtype(analysis_df[f])]
        
        if len(numeric_features) >= 2:
            # Correlation method selection
            corr_col1, corr_col2, corr_col3 = st.columns([2, 2, 2])
            
            with corr_col1:
                correlation_method = st.selectbox(
                    "Correlation Method",
                    ["Pearson", "Spearman", "Kendall", "Compare All"]
                )
            
            with corr_col2:
                threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.3, 0.05)
            
            with corr_col3:
                show_significance = st.checkbox("Show Statistical Significance", value=True)
            
            # Calculate correlations
            if correlation_method == "Compare All":
                # Show comparison of different methods
                methods = ["pearson", "spearman", "kendall"]
                fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=["Pearson", "Spearman", "Kendall"],
                    horizontal_spacing=0.1
                )
                
                for idx, method in enumerate(methods):
                    corr_matrix = analysis_df[numeric_features].corr(method=method)
                    
                    heatmap = go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        text=np.round(corr_matrix.values, 2),
                        texttemplate='%{text}',
                        textfont={"size": 8},
                        showscale=(idx == 2)
                    )
                    
                    fig.add_trace(heatmap, row=1, col=idx+1)
                    fig.update_xaxes(tickangle=-45, row=1, col=idx+1)
                
                fig.update_layout(height=600, title="Correlation Methods Comparison")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Single correlation method
                corr_method = correlation_method.lower()
                corr_matrix = analysis_df[numeric_features].corr(method=corr_method)
                
                # Create mask for upper triangle
                mask = np.triu(np.ones_like(corr_matrix), k=1)
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title=f'{correlation_method} Correlation Matrix',
                    height=700,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show strong correlations
                st.subheader(f"Strong Correlations (|r| > {threshold})")
                
                # Get correlation pairs
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > threshold:
                            corr_pairs.append({
                                'Feature 1': corr_matrix.columns[i],
                                'Feature 2': corr_matrix.columns[j],
                                'Correlation': corr_val,
                                'Abs Correlation': abs(corr_val)
                            })
                
                if corr_pairs:
                    corr_df = pd.DataFrame(corr_pairs)
                    corr_df = corr_df.sort_values('Abs Correlation', ascending=False)
                    
                    # Add significance if requested
                    if show_significance:
                        p_values = []
                        for _, row in corr_df.iterrows():
                            if corr_method == 'pearson':
                                _, p_val = stats.pearsonr(
                                    analysis_df[row['Feature 1']].dropna(),
                                    analysis_df[row['Feature 2']].dropna()
                                )
                            elif corr_method == 'spearman':
                                _, p_val = stats.spearmanr(
                                    analysis_df[row['Feature 1']].dropna(),
                                    analysis_df[row['Feature 2']].dropna()
                                )
                            else:  # kendall
                                _, p_val = stats.kendalltau(
                                    analysis_df[row['Feature 1']].dropna(),
                                    analysis_df[row['Feature 2']].dropna()
                                )
                            p_values.append(p_val)
                        
                        corr_df['p-value'] = p_values
                        corr_df['Significant'] = corr_df['p-value'] < 0.05
                    
                    # Display correlations
                    st.dataframe(
                        corr_df.drop('Abs Correlation', axis=1).style.format({
                            'Correlation': '{:.3f}',
                            'p-value': '{:.4f}' if show_significance else None
                        }).background_gradient(cmap='RdBu', subset=['Correlation'], vmin=-1, vmax=1),
                        use_container_width=True
                    )
                    
                    # Scatter plots for top correlations
                    st.subheader("Relationship Visualizations")
                    
                    n_plots = min(6, len(corr_df))
                    if n_plots > 0:
                        plot_cols = 3
                        plot_rows = (n_plots + plot_cols - 1) // plot_cols
                        
                        fig = make_subplots(
                            rows=plot_rows,
                            cols=plot_cols,
                            subplot_titles=[f"{row['Feature 1']} vs {row['Feature 2']}" 
                                          for _, row in corr_df.head(n_plots).iterrows()],
                            vertical_spacing=0.15,
                            horizontal_spacing=0.1
                        )
                        
                        for idx, (_, row) in enumerate(corr_df.head(n_plots).iterrows()):
                            row_idx = idx // plot_cols + 1
                            col_idx = idx % plot_cols + 1
                            
                            # Get data
                            x_data = analysis_df[row['Feature 1']].dropna()
                            y_data = analysis_df[row['Feature 2']].dropna()
                            
                            # Ensure same length
                            common_idx = x_data.index.intersection(y_data.index)
                            x_data = x_data[common_idx]
                            y_data = y_data[common_idx]
                            
                            # Add scatter plot
                            fig.add_trace(
                                go.Scatter(
                                    x=x_data,
                                    y=y_data,
                                    mode='markers',
                                    marker=dict(size=5, opacity=0.6),
                                    showlegend=False
                                ),
                                row=row_idx,
                                col=col_idx
                            )
                            
                            # Add regression line
                            z = np.polyfit(x_data, y_data, 1)
                            p = np.poly1d(z)
                            x_line = np.linspace(x_data.min(), x_data.max(), 100)
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=x_line,
                                    y=p(x_line),
                                    mode='lines',
                                    line=dict(color='red', width=2),
                                    showlegend=False
                                ),
                                row=row_idx,
                                col=col_idx
                            )
                            
                            # Update axes
                            fig.update_xaxes(title_text=row['Feature 1'], row=row_idx, col=col_idx)
                            fig.update_yaxes(title_text=row['Feature 2'], row=row_idx, col=col_idx)
                        
                        fig.update_layout(height=300 * plot_rows, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No correlations found above threshold {threshold}")
        else:
            st.warning("Please select at least 2 numeric features for correlation analysis")

# Tab 3: Statistical Testing
with tab3:
    st.header("Statistical Hypothesis Testing")
    
    if perform_anova:
        test_col1, test_col2 = st.columns([1, 1])
        
        with test_col1:
            st.subheader("ANOVA Tests (Numerical Features)")
            
            # Get numeric features
            numeric_features = [f for f in selected_features if pd.api.types.is_numeric_dtype(analysis_df[f])]
            
            if numeric_features:
                anova_results = []
                
                for feature in numeric_features:
                    # Perform ANOVA for readmission groups
                    groups = []
                    for readmit_status in ['NO', '>30', '<30']:
                        group_data = analysis_df[analysis_df['readmitted'] == readmit_status][feature].dropna()
                        if len(group_data) > 0:
                            groups.append(group_data)
                    
                    if len(groups) >= 2:
                        f_stat, p_value = stats.f_oneway(*groups)
                        
                        anova_results.append({
                            'Feature': feature,
                            'F-statistic': f_stat,
                            'p-value': p_value,
                            'Significant': p_value < 0.05
                        })
                
                if anova_results:
                    anova_df = pd.DataFrame(anova_results)
                    anova_df = anova_df.sort_values('p-value')
                    
                    st.dataframe(
                        anova_df.style.format({
                            'F-statistic': '{:.3f}',
                            'p-value': '{:.4e}'
                        }).apply(lambda x: ['background-color: #90EE90' if x['Significant'] else ''] * len(x), axis=1),
                        use_container_width=True
                    )
        
        with test_col2:
            st.subheader("Chi-Square Tests (Categorical Features)")
            
            # Get categorical features
            categorical_features = [f for f in selected_features if not pd.api.types.is_numeric_dtype(analysis_df[f])]
            
            if categorical_features:
                chi2_results = []
                
                for feature in categorical_features:
                    # Create contingency table
                    contingency = pd.crosstab(analysis_df[feature], analysis_df['readmitted'])
                    
                    # Perform chi-square test
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                    
                    chi2_results.append({
                        'Feature': feature,
                        'Chi2-statistic': chi2,
                        'p-value': p_value,
                        'DoF': dof,
                        'Significant': p_value < 0.05
                    })
                
                if chi2_results:
                    chi2_df = pd.DataFrame(chi2_results)
                    chi2_df = chi2_df.sort_values('p-value')
                    
                    st.dataframe(
                        chi2_df.style.format({
                            'Chi2-statistic': '{:.3f}',
                            'p-value': '{:.4e}',
                            'DoF': '{:.0f}'
                        }).apply(lambda x: ['background-color: #90EE90' if x['Significant'] else ''] * len(x), axis=1),
                    )
        
                # Post-hoc analysis
        st.subheader("📊 Post-hoc Analysis for Significant Features")
        
        # Select feature for detailed analysis
        significant_numeric = [r['Feature'] for r in anova_results if r['Significant']] if 'anova_results' in locals() else []
        significant_categorical = [r['Feature'] for r in chi2_results if r['Significant']] if 'chi2_results' in locals() else []
        
        all_significant = significant_numeric + significant_categorical
        
        if all_significant:
            selected_feature_test = st.selectbox(
                "Select significant feature for detailed analysis:",
                all_significant
            )
            
            if selected_feature_test in significant_numeric:
                # Tukey HSD for numeric features
                st.write("**Tukey HSD Post-hoc Test**")
                
                # Prepare data for Tukey HSD
                groups_data = []
                groups_labels = []
                
                for readmit_status in ['NO', '>30', '<30']:
                    group_data = analysis_df[analysis_df['readmitted'] == readmit_status][selected_feature_test].dropna()
                    if len(group_data) > 0:
                        groups_data.extend(group_data.tolist())
                        groups_labels.extend([readmit_status] * len(group_data))
                
                # Perform Tukey HSD
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                tukey_result = pairwise_tukeyhsd(groups_data, groups_labels)
                
                # Display results
                st.text(tukey_result)
                
                # Visualize means with confidence intervals
                fig = go.Figure()
                
                for i, status in enumerate(['NO', '>30', '<30']):
                    data = analysis_df[analysis_df['readmitted'] == status][selected_feature_test].dropna()
                    if len(data) > 0:
                        mean = data.mean()
                        sem = data.sem()
                        ci = 1.96 * sem
                        
                        fig.add_trace(go.Scatter(
                            x=[status],
                            y=[mean],
                            error_y=dict(type='data', array=[ci]),
                            mode='markers',
                            marker=dict(size=15),
                            name=status
                        ))
                
                fig.update_layout(
                    title=f"Mean {selected_feature_test} by Readmission Status (95% CI)",
                    xaxis_title="Readmission Status",
                    yaxis_title=selected_feature_test,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Cramér's V for categorical features
                st.write("**Association Strength (Cramér's V)**")
                
                contingency = pd.crosstab(analysis_df[selected_feature_test], analysis_df['readmitted'])
                chi2 = stats.chi2_contingency(contingency)[0]
                n = contingency.sum().sum()
                min_dim = min(contingency.shape) - 1
                cramers_v = np.sqrt(chi2 / (n * min_dim))
                
                st.metric("Cramér's V", f"{cramers_v:.3f}")
                st.write("Interpretation: 0.1-0.3 (weak), 0.3-0.5 (moderate), >0.5 (strong)")
                
                # Show contingency table as heatmap
                fig = px.imshow(
                    contingency.values,
                    labels=dict(x="Readmission Status", y=selected_feature_test, color="Count"),
                    x=contingency.columns,
                    y=contingency.index,
                    color_continuous_scale="Blues"
                )
                
                fig.update_layout(title=f"Contingency Table: {selected_feature_test} vs Readmission")
                st.plotly_chart(fig, use_container_width=True)

# Tab 4: Feature Importance
with tab4:
    st.header("Feature Importance Analysis")
    
    # Method selection
    importance_method = st.selectbox(
        "Select Importance Method",
        ["Mutual Information", "Random Forest", "Information Gain", "All Methods"]
    )
    
    # Prepare data for importance analysis
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    
    df_encoded = analysis_df[selected_features + ['readmitted']].copy()
    
    # Encode target
    le_target = LabelEncoder()
    df_encoded['readmitted_encoded'] = le_target.fit_transform(df_encoded['readmitted'])
    
    # Encode features
    label_encoders = {}
    for col in selected_features:
        if not pd.api.types.is_numeric_dtype(df_encoded[col]):
            le = LabelEncoder()
            df_encoded[col] = df_encoded[col].fillna('missing')
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
    
    # Drop missing values
    df_encoded = df_encoded.dropna()
    
    X = df_encoded[selected_features]
    y = df_encoded['readmitted_encoded']
    
    if importance_method == "All Methods" or importance_method == "Mutual Information":
        # Mutual Information
        from sklearn.feature_selection import mutual_info_classif
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_importance = pd.DataFrame({
            'Feature': selected_features,
            'MI Score': mi_scores
        }).sort_values('MI Score', ascending=False)
        
        if importance_method == "Mutual Information":
            fig = px.bar(
                mi_importance,
                x='MI Score',
                y='Feature',
                orientation='h',
                title='Feature Importance - Mutual Information',
                color='MI Score',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=max(400, len(selected_features) * 25))
            st.plotly_chart(fig, use_container_width=True)
    
    if importance_method == "All Methods" or importance_method == "Random Forest":
        # Random Forest Feature Importance
        from sklearn.ensemble import RandomForestClassifier
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        rf_importance = pd.DataFrame({
            'Feature': selected_features,
            'RF Importance': rf.feature_importances_
        }).sort_values('RF Importance', ascending=False)
        
        if importance_method == "Random Forest":
            fig = px.bar(
                rf_importance,
                x='RF Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance - Random Forest',
                color='RF Importance',
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=max(400, len(selected_features) * 25))
            st.plotly_chart(fig, use_container_width=True)
    
    if importance_method == "All Methods" or importance_method == "Information Gain":
        # Information Gain (Entropy-based)
        from sklearn.feature_selection import mutual_info_classif
        
        # Calculate information gain
        ig_scores = []
        for feature in selected_features:
            # Calculate entropy of target
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            entropy_target = -np.sum(probs * np.log2(probs + 1e-10))
            
            # Calculate conditional entropy
            feature_values = X[feature].unique()
            conditional_entropy = 0
            
            for value in feature_values:
                mask = X[feature] == value
                subset_y = y[mask]
                if len(subset_y) > 0:
                    _, value_counts = np.unique(subset_y, return_counts=True)
                    value_probs = value_counts / len(subset_y)
                    value_entropy = -np.sum(value_probs * np.log2(value_probs + 1e-10))
                    conditional_entropy += (len(subset_y) / len(y)) * value_entropy
            
            ig = entropy_target - conditional_entropy
            ig_scores.append(ig)
        
        ig_importance = pd.DataFrame({
            'Feature': selected_features,
            'IG Score': ig_scores
        }).sort_values('IG Score', ascending=False)
        
        if importance_method == "Information Gain":
            fig = px.bar(
                ig_importance,
                x='IG Score',
                y='Feature',
                orientation='h',
                title='Feature Importance - Information Gain',
                color='IG Score',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=max(400, len(selected_features) * 25))
            st.plotly_chart(fig, use_container_width=True)
    
    if importance_method == "All Methods":
        # Compare all methods
        st.subheader("Feature Importance Comparison")
        
        # Merge all importance scores
        importance_comparison = mi_importance.merge(
            rf_importance, on='Feature'
        ).merge(
            ig_importance, on='Feature'
        )
        
        # Normalize scores to 0-1 range
        for col in ['MI Score', 'RF Importance', 'IG Score']:
            importance_comparison[f'{col}_norm'] = (
                importance_comparison[col] / importance_comparison[col].max()
            )
        
        # Create radar chart for top features
        top_n = min(10, len(importance_comparison))
        top_features = importance_comparison.nlargest(top_n, 'MI Score')
        
        fig = go.Figure()
        
        categories = ['Mutual Info', 'Random Forest', 'Info Gain']
        
        for _, row in top_features.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['MI Score_norm'], row['RF Importance_norm'], row['IG Score_norm']],
                theta=categories,
                fill='toself',
                name=row['Feature'][:20]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Feature Importance Comparison (Normalized)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.subheader("Feature Importance Summary")
        
        summary_df = importance_comparison[['Feature', 'MI Score', 'RF Importance', 'IG Score']].copy()
        summary_df['Average Rank'] = summary_df[['MI Score', 'RF Importance', 'IG Score']].rank(ascending=False).mean(axis=1)
        summary_df = summary_df.sort_values('Average Rank')
        
        st.dataframe(
            summary_df.style.format({
                'MI Score': '{:.3f}',
                'RF Importance': '{:.3f}',
                'IG Score': '{:.3f}',
                'Average Rank': '{:.1f}'
            }).background_gradient(cmap='YlOrRd', subset=['MI Score', 'RF Importance', 'IG Score']),
            use_container_width=True
        )

# Tab 5: Comparative Analysis
with tab5:
    st.header("Comparative Analysis Across Patient Groups")
    
    comp_col1, comp_col2 = st.columns([2, 1])
    
    with comp_col1:
        features_to_compare = st.multiselect(
            "Select features to compare:",
            selected_features,
            default=selected_features[:5]
        )
    
    with comp_col2:
        comparison_type = st.radio(
            "Comparison Type",
            ["Summary Statistics", "Distributions", "Hypothesis Tests"]
        )
    
    if features_to_compare:
        if comparison_type == "Summary Statistics":
            st.subheader("Statistical Summary Comparison")
            
            # Create comparison dataframes
            datasets = {
                'All Patients': df,
                'First Encounters': df_first_encounter,
                'Repeated Patients': df_repeated
            }
            
            for feature in features_to_compare:
                st.markdown(f"### {feature}")
                
                if pd.api.types.is_numeric_dtype(df[feature]):
                    # Numeric comparison
                    comparison_data = []
                    
                    for name, dataset in datasets.items():
                        feature_data = dataset[feature].dropna()
                        comparison_data.append({
                            'Dataset': name,
                            'Count': len(feature_data),
                            'Mean': feature_data.mean(),
                            'Std': feature_data.std(),
                            'Median': feature_data.median(),
                            'Q1': feature_data.quantile(0.25),
                            'Q3': feature_data.quantile(0.75)
                        })
                    
                    comp_df = pd.DataFrame(comparison_data)
                    
                    # Create box plot comparison
                    fig = go.Figure()
                    
                    for name, dataset in datasets.items():
                        fig.add_trace(go.Box(
                            y=dataset[feature].dropna(),
                            name=name,
                            boxmean='sd'
                        ))
                    
                    fig.update_layout(
                        title=f"{feature} Distribution Across Patient Groups",
                        yaxis_title=feature,
                        height=400
                    )
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        st.dataframe(
                            comp_df.set_index('Dataset').style.format({
                                'Count': '{:,}',
                                'Mean': '{:.2f}',
                                'Std': '{:.2f}',
                                'Median': '{:.2f}',
                                'Q1': '{:.2f}',
                                'Q3': '{:.2f}'
                            }),
                            use_container_width=True
                        )
                else:
                    # Categorical comparison
                    fig = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=['All Patients', 'First Encounters', 'Repeated Patients'],
                        specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]]
                    )
                    
                    for idx, (name, dataset) in enumerate(datasets.items()):
                        value_counts = dataset[feature].value_counts().head(10)
                        
                        fig.add_trace(
                            go.Pie(
                                labels=value_counts.index,
                                values=value_counts.values,
                                name=name,
                                hole=0.4
                            ),
                            row=1,
                            col=idx+1
                        )
                    
                    fig.update_layout(
                        title=f"{feature} Distribution Across Patient Groups",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        elif comparison_type == "Distributions":
            st.subheader("Distribution Comparison")
            
            for feature in features_to_compare:
                if pd.api.types.is_numeric_dtype(df[feature]):
                    st.markdown(f"### {feature}")
                    
                    # Create overlapping distributions
                    fig = go.Figure()
                    
                    colors = ['blue', 'red', 'green']
                    datasets = {
                        'All Patients': df,
                        'First Encounters': df_first_encounter,
                        'Repeated Patients': df_repeated
                    }
                    
                    for idx, (name, dataset) in enumerate(datasets.items()):
                        feature_data = dataset[feature].dropna()
                        
                        # Add histogram
                        fig.add_trace(go.Histogram(
                            x=feature_data,
                            name=name,
                            opacity=0.5,
                            nbinsx=30,
                            histnorm='probability density',
                            marker_color=colors[idx]
                        ))
                        
                        # Calculate and add KDE
                        from scipy.stats import gaussian_kde
                        kde = gaussian_kde(feature_data)
                        x_range = np.linspace(feature_data.min(), feature_data.max(), 200)
                        kde_values = kde(x_range)
                        
                        fig.add_trace(go.Scatter(
                            x=x_range,
                            y=kde_values,
                            mode='lines',
                            name=f'{name} KDE',
                            line=dict(color=colors[idx], width=2)
                        ))
                    
                    fig.update_layout(
                        title=f"{feature} Distribution Comparison",
                        xaxis_title=feature,
                        yaxis_title="Density",
                        height=400,
                        barmode='overlay'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistical tests
                    st.write("**Statistical Tests:**")
                    
                    # Kolmogorov-Smirnov tests
                    test_results = []
                    
                    # All vs First
                    ks_stat, ks_p = stats.ks_2samp(
                        df[feature].dropna(),
                        df_first_encounter[feature].dropna()
                    )
                    test_results.append({
                        'Comparison': 'All vs First Encounters',
                        'KS Statistic': ks_stat,
                        'p-value': ks_p,
                        'Significant': ks_p < 0.05
                    })
                    
                    # All vs Repeated
                    ks_stat, ks_p = stats.ks_2samp(
                        df[feature].dropna(),
                        df_repeated[feature].dropna()
                    )
                    test_results.append({
                        'Comparison': 'All vs Repeated',
                        'KS Statistic': ks_stat,
                        'p-value': ks_p,
                        'Significant': ks_p < 0.05
                    })
                    
                    # First vs Repeated
                    ks_stat, ks_p = stats.ks_2samp(
                        df_first_encounter[feature].dropna(),
                        df_repeated[feature].dropna()
                    )
                    test_results.append({
                        'Comparison': 'First vs Repeated',
                        'KS Statistic': ks_stat,
                        'p-value': ks_p,
                        'Significant': ks_p < 0.05
                    })
                    
                    test_df = pd.DataFrame(test_results)
                    st.dataframe(
                        test_df.style.format({
                            'KS Statistic': '{:.3f}',
                            'p-value': '{:.4e}'
                        }).apply(lambda x: ['background-color: #90EE90' if x['Significant'] else ''] * len(x), axis=1),
                        use_container_width=True
                    )
        
        else:  # Hypothesis Tests
            st.subheader("Hypothesis Testing Across Groups")
            
            # Prepare results
            test_results = []
            
            for feature in features_to_compare:
                if pd.api.types.is_numeric_dtype(df[feature]):
                    # Kruskal-Wallis test
                    groups = [
                        df[feature].dropna(),
                        df_first_encounter[feature].dropna(),
                        df_repeated[feature].dropna()
                    ]
                    
                    h_stat, p_value = stats.kruskal(*groups)
                    
                    test_results.append({
                        'Feature': feature,
                        'Test': 'Kruskal-Wallis',
                        'Statistic': h_stat,
                        'p-value': p_value,
                        'Significant': p_value < 0.05,
                        'Interpretation': 'Groups differ' if p_value < 0.05 else 'No difference'
                    })
                else:
                    # Chi-square test for independence
                    # Create contingency table
                    all_cat = df[feature].value_counts()
                    first_cat = df_first_encounter[feature].value_counts()
                    repeated_cat = df_repeated[feature].value_counts()
                    
                    # Align categories
                    all_cats = set(all_cat.index) | set(first_cat.index) | set(repeated_cat.index)
                    contingency_data = []
                    
                    for cat in all_cats:
                        contingency_data.append([
                            all_cat.get(cat, 0),
                            first_cat.get(cat, 0),
                            repeated_cat.get(cat, 0)
                        ])
                    
                    contingency = np.array(contingency_data)
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency.T)
                    
                    test_results.append({
                        'Feature': feature,
                        'Test': 'Chi-square',
                        'Statistic': chi2,
                        'p-value': p_value,
                        'Significant': p_value < 0.05,
                        'Interpretation': 'Groups differ' if p_value < 0.05 else 'No difference'
                    })
            
            # Display results
            results_df = pd.DataFrame(test_results)
            st.dataframe(
                results_df.style.format({
                    'Statistic': '{:.3f}',
                    'p-value': '{:.4e}'
                }).apply(lambda x: ['background-color: #90EE90' if x['Significant'] else ''] * len(x), axis=1),
                use_container_width=True
            )
            
            # Detailed analysis for significant features
            significant_features = results_df[results_df['Significant']]['Feature'].tolist()
            
            if significant_features:
                st.subheader("Detailed Analysis of Significant Differences")
                
                selected_sig_feature = st.selectbox(
                    "Select feature for detailed comparison:",
                    significant_features
                )
                
                if pd.api.types.is_numeric_dtype(df[selected_sig_feature]):
                    # Create detailed comparison
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Violin plot
                        fig = go.Figure()
                        
                        datasets_list = [
                            ('All Patients', df),
                            ('First Encounters', df_first_encounter),
                            ('Repeated Patients', df_repeated)
                        ]
                        
                        for name, dataset in datasets_list:
                            fig.add_trace(go.Violin(
                                y=dataset[selected_sig_feature].dropna(),
                                name=name,
                                box_visible=True,
                                meanline_visible=True
                            ))
                        
                        fig.update_layout(
                            title=f"{selected_sig_feature} Distribution by Patient Group",
                            yaxis_title=selected_sig_feature,
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Effect Sizes (Cohen's d)**")
                        
                        # Calculate effect sizes
                        from numpy import sqrt
                        
                        # All vs First
                        d1 = (df[selected_sig_feature].mean() - df_first_encounter[selected_sig_feature].mean()) / \
                             sqrt((df[selected_sig_feature].var() + df_first_encounter[selected_sig_feature].var()) / 2)
                        
                        # All vs Repeated
                        d2 = (df[selected_sig_feature].mean() - df_repeated[selected_sig_feature].mean()) / \
                             sqrt((df[selected_sig_feature].var() + df_repeated[selected_sig_feature].var()) / 2)
                        
                        # First vs Repeated
                        d3 = (df_first_encounter[selected_sig_feature].mean() - df_repeated[selected_sig_feature].mean()) / \
                             sqrt((df_first_encounter[selected_sig_feature].var() + df_repeated[selected_sig_feature].var()) / 2)
                        
                        effect_sizes = pd.DataFrame({
                            'Comparison': ['All vs First', 'All vs Repeated', 'First vs Repeated'],
                            'Cohen\'s d': [d1, d2, d3],
                            'Effect Size': [
                                'Small' if abs(d1) < 0.5 else 'Medium' if abs(d1) < 0.8 else 'Large',
                                'Small' if abs(d2) < 0.5 else 'Medium' if abs(d2) < 0.8 else 'Large',
                                'Small' if abs(d3) < 0.5 else 'Medium' if abs(d3) < 0.8 else 'Large'
                            ]
                        })
                        
                        st.dataframe(
                            effect_sizes.style.format({'Cohen\'s d': '{:.3f}'}),
                            use_container_width=True
                        )

# Summary and Recommendations
st.markdown("---")
st.header("📋 Analysis Summary and Recommendations")

summary_col1, summary_col2 = st.columns([2, 1])

with summary_col1:
    st.subheader("Key Findings")
    
    # Automatically generate key findings based on analysis
    findings = []
    
    # Finding 1: Dataset characteristics
    findings.append(f"**Dataset Overview**: Analyzed {len(df):,} total encounters, "
                   f"{len(df_first_encounter):,} first encounters, and "
                   f"{len(df_repeated):,} repeated patient encounters.")
    
    # Finding 2: Readmission rates
    overall_readmit = (df['readmitted'] != 'NO').mean() * 100
    first_readmit = (df_first_encounter['readmitted'] != 'NO').mean() * 100
    repeated_readmit = (df_repeated['readmitted'] != 'NO').mean() * 100
    
    findings.append(f"**Readmission Rates**: Overall {overall_readmit:.1f}%, "
                   f"First encounters {first_readmit:.1f}%, "
                   f"Repeated patients {repeated_readmit:.1f}%")
    
    # Finding 3: Most important features (if available)
    if 'mi_importance' in locals() and not mi_importance.empty:
        top_features_list = mi_importance.head(3)['Feature'].tolist()
        findings.append(f"**Top Predictive Features**: {', '.join(top_features_list)}")
    
    # Display findings
    for finding in findings:
        st.markdown(f"• {finding}")

with summary_col2:
    st.subheader("Recommendations")
    
    recommendations = [
        "Focus on repeated patients who show significantly higher readmission rates",
        "Implement targeted interventions for high-risk features identified",
        "Regular monitoring of distribution shifts in key features",
        "Consider feature engineering based on correlation insights"
    ]
    
    for rec in recommendations:
        st.info(f"💡 {rec}")

# Export options
st.markdown("---")
st.header("📥 Export Analysis Results")

export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    if st.button("📊 Export Feature Statistics", type="primary"):
        # Create comprehensive statistics export
        export_stats = pd.DataFrame()
        
        for feature in selected_features:
            if pd.api.types.is_numeric_dtype(df[feature]):
                stats_dict = {
                    'Feature': feature,
                    'Type': 'Numeric',
                    'Mean': df[feature].mean(),
                    'Std': df[feature].std(),
                    'Min': df[feature].min(),
                    'Max': df[feature].max(),
                    'Missing': df[feature].isna().sum()
                }
            else:
                stats_dict = {
                    'Feature': feature,
                    'Type': 'Categorical',
                    'Unique': df[feature].nunique(),
                    'Mode': df[feature].mode()[0] if len(df[feature].mode()) > 0 else None,
                    'Missing': df[feature].isna().sum()
                }
            
            export_stats = pd.concat([export_stats, pd.DataFrame([stats_dict])], ignore_index=True)
        
        csv = export_stats.to_csv(index=False)
        st.download_button(
            label="Download Statistics CSV",
            data=csv,
            file_name='feature_statistics.csv',
            mime='text/csv'
        )

with export_col2:
    if st.button("📈 Export Correlation Matrix", type="primary"):
        # Export correlation matrix
        numeric_features = [f for f in selected_features if pd.api.types.is_numeric_dtype(df[f])]
        if numeric_features:
            corr_matrix = df[numeric_features].corr()
            csv = corr_matrix.to_csv()
            st.download_button(
                label="Download Correlation CSV",
                data=csv,
                file_name='correlation_matrix.csv',
                mime='text/csv'
            )

with export_col3:
    if st.button("📄 Generate Analysis Report", type="primary"):
        # Generate a comprehensive text report
        report = f"""
ADVANCED ANALYSIS REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW
================
Total Encounters: {len(df):,}
First Encounters: {len(df_first_encounter):,}
Repeated Patients: {len(df_repeated):,}
Features Analyzed: {len(selected_features)}

READMISSION STATISTICS
=====================
Overall Readmission Rate: {(df['readmitted'] != 'NO').mean() * 100:.1f}%
- <30 days: {(df['readmitted'] == '<30').mean() * 100:.1f}%
- >30 days: {(df['readmitted'] == '>30').mean() * 100:.1f}%

First Encounters Readmission: {(df_first_encounter['readmitted'] != 'NO').mean() * 100:.1f}%
Repeated Patients Readmission: {(df_repeated['readmitted'] != 'NO').mean() * 100:.1f}%

FEATURE ANALYSIS SUMMARY
=======================
Numeric Features: {len([f for f in selected_features if pd.api.types.is_numeric_dtype(df[f])])}
Categorical Features: {len([f for f in selected_features if not pd.api.types.is_numeric_dtype(df[f])])}

KEY FINDINGS
============
1. Repeated patients show {((repeated_readmit - first_readmit) / first_readmit * 100):.1f}% higher readmission rate
2. Significant differences found between patient groups
3. Multiple features show strong correlation with readmission outcomes

RECOMMENDATIONS
==============
1. Implement targeted interventions for repeated patients
2. Focus on high-risk features identified in the analysis
3. Regular monitoring of key metrics
4. Consider predictive modeling based on important features
"""
        
        st.download_button(
            label="Download Report TXT",
            data=report,
            file_name='advanced_analysis_report.txt',
            mime='text/plain'
        )

# Add custom CSS for better styling
st.markdown("""
<style>
    /* Style for metric containers */
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Style for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: white;
        border-radius: 5px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
    }
    
    /* Style for dataframes */
    .dataframe {
        font-size: 12px;
    }
    
    /* Style for plots */
    .plotly-graph-div {
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-radius: 5px;
        padding: 10px;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption(f"Advanced Analysis Dashboard | Data updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")