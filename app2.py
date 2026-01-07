import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import joblib

# Page config
st.set_page_config(
    page_title="Clinical Trial Data Quality Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('outputs/all_studies_noisy.csv')
    anomalies = pd.read_csv('outputs/anomaly_results.csv')
    study_comparison = pd.read_csv('outputs/study_performance_comparison.csv')
    return df, anomalies, study_comparison

df, anomalies, study_comparison = load_data()

# Sidebar
st.sidebar.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=Novartis+NEST", use_container_width=True)
st.sidebar.title("🏥 Clinical Trial DQI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Executive Dashboard", "📊 Multi-Study Intelligence", "🔍 Study Deep Dive", 
     "🤖 AI Models", "🚨 Anomalies", "📈 Insights & ROI", "🧠 Gen-AI Assistant","⚠️ Early Warning Predictor"]
)

# Main title
st.markdown('<p class="main-header">🏥 Clinical Trial Data Quality Intelligence Platform</p>', unsafe_allow_html=True)
st.markdown("**AI-Powered Analytics for 16 Clinical Trials | 6,237 Patients**")
st.markdown("---")

# ============================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# ============================================================================
if page == "🏠 Executive Dashboard":
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    avg_dqi = df['Smart_DQI_Noisy'].mean()
    total_patients = len(df)
    high_risk_count = (df['Smart_Risk_Noisy'] == 'High').sum()
    anomaly_count = anomalies['Is_Anomaly'].sum()
    
    with col1:
        st.metric("📊 Average DQI", f"{avg_dqi:.1f}", 
                 delta=f"{avg_dqi - 85:.1f} vs target" if avg_dqi >= 85 else f"{avg_dqi - 85:.1f} vs target",
                 delta_color="normal" if avg_dqi >= 85 else "inverse")
    
    with col2:
        st.metric("👥 Total Patients", f"{total_patients:,}", "Across 16 studies")
    
    with col3:
        high_risk_pct = high_risk_count / total_patients * 100
        st.metric("🚨 High Risk Patients", f"{high_risk_count:,}", 
                 f"{high_risk_pct:.1f}% of total",
                 delta_color="inverse")
    
    with col4:
        anomaly_pct = anomaly_count / total_patients * 100
        st.metric("🔍 Anomalies Detected", f"{anomaly_count:,}", 
                 f"{anomaly_pct:.1f}% flagged")
    
    st.markdown("---")
    
    # Two columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Risk Distribution")
        risk_counts = df['Smart_Risk_Noisy'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'},
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Smart DQI by Study")
        study_dqi = df.groupby('Study')['Smart_DQI_Noisy'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=study_dqi.values,
            y=[s[:30] for s in study_dqi.index],
            orientation='h',
            color=study_dqi.values,
            color_continuous_scale=['#dc3545', '#ffc107', '#28a745'],
            range_color=[0, 100]
        )
        fig.update_layout(
            height=400,
            xaxis_title="Smart DQI Score",
            yaxis_title="Study",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Bottom row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Top 5 Problem Areas")
        problem_studies = df.groupby('Study').agg({
            'Smart_DQI_Noisy': 'mean',
            'SAE_Pending_Count': 'sum',
            'Overdue_Visits_Count': 'sum'
        }).nsmallest(5, 'Smart_DQI_Noisy')
        
        for idx, (study, row) in enumerate(problem_studies.iterrows(), 1):
            st.markdown(f"""
            <div class="{'danger-box' if idx <= 2 else 'warning-box'}">
                <strong>{idx}. {study[:40]}</strong><br>
                DQI: {row['Smart_DQI_Noisy']:.1f} | 
                SAE Pending: {int(row['SAE_Pending_Count'])} | 
                Overdue Visits: {int(row['Overdue_Visits_Count'])}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
    
    with col2:
        st.subheader("⭐ Top 5 Performing Studies")
        top_studies = df.groupby('Study').agg({
            'Smart_DQI_Noisy': 'mean',
            'Subject ID': 'count'
        }).nlargest(5, 'Smart_DQI_Noisy')
        
        for idx, (study, row) in enumerate(top_studies.iterrows(), 1):
            st.markdown(f"""
            <div class="success-box">
                <strong>{idx}. {study[:40]}</strong><br>
                DQI: {row['Smart_DQI_Noisy']:.1f} | 
                Patients: {int(row['Subject ID'])}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# PAGE 2: MULTI-STUDY INTELLIGENCE
# ============================================================================
elif page == "📊 Multi-Study Intelligence":
    
    st.header("🧠 Cross-Study Learning & Best Practices")
    st.markdown("**Identify what top-performing studies do differently and transfer knowledge**")
    st.markdown("---")
    
    # Study performance comparison
    study_stats = study_comparison.copy()
    # Check if 'Study' is already a column
    if 'Study' not in study_stats.columns:
      study_stats = study_stats.reset_index()
    
    # Best and worst
    best_study = study_stats.loc[study_stats['Avg_DQI'].idxmax()]
    worst_study = study_stats.loc[study_stats['Avg_DQI'].idxmin()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="success-box">
            <h3>⭐ BEST PERFORMING STUDY</h3>
            <h4>{best_study['Study'][:40]}</h4>
            <p><strong>Average DQI:</strong> {best_study['Avg_DQI']:.1f}</p>
            <p><strong>Total Patients:</strong> {int(best_study['Total_Patients'])}</p>
            <p><strong>High Risk:</strong> {best_study['High_Risk_Pct']:.1f}%</p>
            <p><strong>Avg SAE Pending:</strong> {best_study['Avg_SAE']:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="danger-box">
            <h3>⚠️ NEEDS IMPROVEMENT</h3>
            <h4>{worst_study['Study'][:40]}</h4>
            <p><strong>Average DQI:</strong> {worst_study['Avg_DQI']:.1f}</p>
            <p><strong>Total Patients:</strong> {int(worst_study['Total_Patients'])}</p>
            <p><strong>High Risk:</strong> {worst_study['High_Risk_Pct']:.1f}%</p>
            <p><strong>Avg SAE Pending:</strong> {worst_study['Avg_SAE']:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Performance gap visualization
    st.subheader("📊 Performance Gap Analysis")
    
    dqi_gap = best_study['Avg_DQI'] - worst_study['Avg_DQI']
    sae_gap = worst_study['Avg_SAE'] - best_study['Avg_SAE']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("DQI Gap", f"{dqi_gap:.1f} points", "Improvement opportunity")
    with col2:
        st.metric("SAE Difference", f"{sae_gap:.1f} fewer", "In best study")
    with col3:
        est_improvement = min(dqi_gap * 0.7, 20)
        st.metric("Potential Gain", f"+{est_improvement:.0f} DQI pts", "If practices adopted")
    
    # Comparative chart
    comparison_metrics = pd.DataFrame({
        'Metric': ['DQI Score', 'SAE Pending', 'Overdue Visits', 'High Risk %'],
        best_study['Study'][:20]: [
            best_study['Avg_DQI'],
            best_study['Avg_SAE'],
            best_study['Avg_Overdue_Visits'],
            best_study['High_Risk_Pct']
        ],
        worst_study['Study'][:20]: [
            worst_study['Avg_DQI'],
            worst_study['Avg_SAE'],
            worst_study['Avg_Overdue_Visits'],
            worst_study['High_Risk_Pct']
        ]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=best_study['Study'][:20],
        x=comparison_metrics['Metric'],
        y=comparison_metrics[best_study['Study'][:20]],
        marker_color='#28a745'
    ))
    fig.add_trace(go.Bar(
        name=worst_study['Study'][:20],
        x=comparison_metrics['Metric'],
        y=comparison_metrics[worst_study['Study'][:20]],
        marker_color='#dc3545'
    ))
    
    fig.update_layout(
        title="Best vs Worst Performing Study Comparison",
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("💡 AI-Generated Recommendations")
    
    if worst_study['Avg_SAE'] > 3:
        sae_improvement = min(15, sae_gap * 2)
        st.markdown(f"""
        <div class="danger-box">
            <h4>🚨 URGENT: Implement Automated SAE Escalation</h4>
            <p><strong>Current:</strong> {worst_study['Avg_SAE']:.1f} avg pending SAE per patient</p>
            <p><strong>Target:</strong> {best_study['Avg_SAE']:.1f} (match best study)</p>
            <p><strong>Action:</strong> Deploy automated alerts for SAE >3 days old</p>
            <p><strong>Expected Impact:</strong> +{sae_improvement:.0f} DQI points</p>
            <p><strong>Timeline:</strong> 2-3 weeks implementation</p>
        </div>
        """, unsafe_allow_html=True)
    
    if worst_study['High_Risk_Pct'] > 15:
        st.markdown(f"""
        <div class="warning-box">
            <h4>⚠️ HIGH PRIORITY: Increase Site Monitoring</h4>
            <p><strong>Current:</strong> {worst_study['High_Risk_Pct']:.1f}% high-risk patients</p>
            <p><strong>Target:</strong> <10% (industry standard)</p>
            <p><strong>Action:</strong> Assign 2 additional CRAs to highest-risk sites</p>
            <p><strong>Expected Impact:</strong> Reduce high-risk by 50% in 4 weeks</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ROI Calculation
    st.markdown("---")
    st.subheader("💰 ROI Projection")
    
    weeks_to_improve = 4
    cost_per_week = 20000
    revenue_per_day = 50000
    days_saved = int(est_improvement * 0.5)
    roi = (days_saved * revenue_per_day) - (weeks_to_improve * cost_per_week)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Implementation Cost", f"${weeks_to_improve * cost_per_week:,}", f"{weeks_to_improve} weeks")
    with col2:
        st.metric("Expected DQI Gain", f"+{est_improvement:.0f} points", "70% of gap closed")
    with col3:
        st.metric("Time Saved", f"{days_saved} days", "Faster database lock")
    with col4:
        st.metric("NET ROI", f"${roi:,}", f"{roi/(weeks_to_improve * cost_per_week)*100:.0f}% return")

# ============================================================================
# PAGE 3: STUDY DEEP DIVE
# ============================================================================
elif page == "🔍 Study Deep Dive":
    
    st.header("🔍 Individual Study Analysis")
    
    # Study selector
    study_list = sorted(df['Study'].unique())
    selected_study = st.selectbox("Select a study to analyze:", study_list)
    
    study_df = df[df['Study'] == selected_study]
    
    st.markdown("---")
    
    # Study metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", len(study_df))
    with col2:
        st.metric("Average DQI", f"{study_df['Smart_DQI_Noisy'].mean():.1f}")
    with col3:
        high_risk = (study_df['Smart_Risk_Noisy'] == 'High').sum()
        st.metric("High Risk", f"{high_risk} ({high_risk/len(study_df)*100:.1f}%)")
    with col4:
        st.metric("Avg SAE Pending", f"{study_df['SAE_Pending_Count'].mean():.1f}")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("DQI Distribution")
        fig = px.histogram(
            study_df,
            x='Smart_DQI_Noisy',
            nbins=20,
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(
            xaxis_title="DQI Score",
            yaxis_title="Number of Patients",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Categories")
        risk_dist = study_df['Smart_Risk_Noisy'].value_counts()
        fig = px.bar(
            x=risk_dist.index,
            y=risk_dist.values,
            color=risk_dist.index,
            color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
        )
        fig.update_layout(
            xaxis_title="Risk Level",
            yaxis_title="Number of Patients",
            showlegend=False,
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Patient table
    st.subheader("📋 Patient Details")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.multiselect("Filter by Risk:", ['Low', 'Medium', 'High'], default=['High', 'Medium'])
    with col2:
        min_sae = st.number_input("Min SAE Pending:", min_value=0, value=0)
    with col3:
        show_top = st.number_input("Show top N patients:", min_value=5, max_value=100, value=20)
    
    filtered_df = study_df[study_df['Smart_Risk_Noisy'].isin(risk_filter)]
    filtered_df = filtered_df[filtered_df['SAE_Pending_Count'] >= min_sae]
    filtered_df = filtered_df.nsmallest(show_top, 'Smart_DQI_Noisy')
    
    display_df = filtered_df[[
        'Subject ID', 'Site ID', 'Smart_DQI_Noisy', 'Smart_Risk_Noisy',
        'SAE_Pending_Count', 'Overdue_Visits_Count', 'Missing_Pages'
    ]].copy()
    display_df.columns = ['Patient ID', 'Site', 'DQI', 'Risk', 'SAE', 'Overdue Visits', 'Missing Pages']
    
    st.dataframe(display_df, use_container_width=True, height=400)

# ============================================================================
# PAGE 4: AI MODELS
# ============================================================================
elif page == "🤖 AI Models":
    
    st.header("🤖 AI/ML Models Performance")
    
    tab1, tab2, tab3 = st.tabs(["📈 Risk Prediction", "🎯 Priority Scoring", "🔍 Anomaly Detection"])
    
    with tab1:
        st.subheader("Model 1: Future Risk Deterioration Predictor")
        st.markdown("**Predicts which patients will develop severe data quality issues**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", "85.5%", "On test set")
        with col2:
            st.metric("ROC-AUC", "0.801", "Good predictive power")
        with col3:
            st.metric("Recall", "35%", "Catches 1 in 3 issues")
        
        st.markdown("---")
        
        st.markdown("""
        **Top Predictive Features:**
        1. **SAE_Pending_Count** (48.4%) - Safety issues are strongest predictor
        2. **Total_Issues** (37.4%) - Overall issue count matters
        3. **Issues_Trend** (3.9%) - Trend direction helps prediction
        4. **Context_Encoded** (2.5%) - Study type influences risk
        
        **Use Case:** Identify patients likely to deteriorate in next 7 days for proactive intervention
        """)
    
    with tab2:
        st.subheader("Model 2: Priority Scoring Engine")
        st.markdown("**Ranks urgency of issues to optimize resource allocation**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", "0.996", "Excellent fit")
        with col2:
            st.metric("MAE", "±1.2 points", "Very accurate")
        with col3:
            st.metric("RMSE", "±1.6 points", "Consistent")
        
        st.markdown("---")
        
        st.markdown("""
        **Top Priority Drivers:**
        1. **Complexity_Score** (89.4%) - Overall complexity dominates
        2. **SAE_squared** (9.0%) - Non-linear SAE impact
        3. **SAE_Emergency** (1.2%) - Critical safety flags
        
        **Priority Categories:**
        - 🔴 URGENT (0-49): 7.7% of patients - Need immediate action
        - 🟡 MODERATE (50-79): 3.5% - Action needed this week  
        - 🟢 LOW (80-100): 88.8% - Routine monitoring
        """)
    
    with tab3:
        st.subheader("Model 3: Anomaly Detection System")
        st.markdown("**Identifies unusual data patterns using Isolation Forest**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Anomalies Detected", "301", "4.8% of patients")
        with col2:
            st.metric("Most Affected Study", "Study 4", "14.5% anomaly rate")
        with col3:
            st.metric("Avg DQI of Anomalies", "63.3", "vs 87.4 normal")
        
        st.markdown("---")
        
        st.markdown("""
        **Anomaly Characteristics:**
        - Average SAE: **17.3** (vs 0.8 normal) - 16.5x higher!
        - Average DQI: **63.3** (vs 87.4 normal) - 24 points lower
        - Average Total Issues: **18.8** (vs 1.1 normal)
        
        **Key Finding:** Study 4 accounts for 75% of all detected anomalies
        """)

# ============================================================================
# PAGE 5: ANOMALIES
# ============================================================================
elif page == "🚨 Anomalies":
    
    st.header("🔍 Detected Anomalies & Outliers")
    st.markdown("**Patients flagged by ML for unusual data patterns**")
    st.markdown("---")
    
    anomaly_df = df.merge(anomalies[['Subject ID', 'Is_Anomaly', 'Anomaly_Score']], 
                          on='Subject ID', how='left')
    anomaly_patients = anomaly_df[anomaly_df['Is_Anomaly'] == 1].copy()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Anomalies", len(anomaly_patients))
    with col2:
        st.metric("Avg DQI", f"{anomaly_patients['Smart_DQI_Noisy'].mean():.1f}")
    with col3:
        st.metric("Avg SAE", f"{anomaly_patients['SAE_Pending_Count'].mean():.1f}")
    with col4:
        high_risk_anom = (anomaly_patients['Smart_Risk_Noisy'] == 'High').sum()
        st.metric("High Risk", f"{high_risk_anom} ({high_risk_anom/len(anomaly_patients)*100:.1f}%)")
    
    st.markdown("---")
    
    # Anomalies by study
    st.subheader("📊 Anomaly Distribution by Study")
    anom_by_study = anomaly_patients['Study'].value_counts().head(10)
    
    fig = px.bar(
        x=anom_by_study.values,
        y=[s[:30] for s in anom_by_study.index],
        orientation='h',
        color=anom_by_study.values,
        color_continuous_scale='Reds'
    )
    fig.update_layout(
        xaxis_title="Number of Anomalies",
        yaxis_title="Study",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top anomalies table
    st.subheader("🚨 Most Severe Anomalies")
    
    top_anomalies = anomaly_patients.nlargest(20, 'Anomaly_Score')[[
        'Subject ID', 'Study', 'Smart_DQI_Noisy', 'Smart_Risk_Noisy',
        'SAE_Pending_Count', 'Overdue_Visits_Count', 'Anomaly_Score'
    ]].copy()
    
    top_anomalies.columns = ['Patient ID', 'Study', 'DQI', 'Risk', 'SAE', 'Overdue Visits', 'Anomaly Score']
    top_anomalies['Study'] = top_anomalies['Study'].str[:35]
    
    st.dataframe(top_anomalies, use_container_width=True, height=500)

# ============================================================================
# PAGE 6: INSIGHTS & ROI
# ============================================================================
elif page == "📈 Insights & ROI":
    
    st.header("📈 Executive Insights & Business Impact")
    
    # Key insights
    st.subheader("🎯 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="danger-box">
            <h4>🚨 Critical Issues Identified</h4>
            <ul>
                <li><strong>Study 4:</strong> 49.1% high-risk patients (763 patients)</li>
                <li><strong>Avg 6.4 pending SAE</strong> per patient - regulatory risk!</li>
                <li><strong>DQI: 74.1</strong> - 17.4 points below best performer</li>
                <li><strong>226 anomalies</strong> detected (14.5% of study)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-box">
            <h4>✅ Success Stories</h4>
            <ul>
                <li><strong>Study 14:</strong> 91.5 DQI - excellence standard</li>
                <li><strong>0 pending SAE</strong> - exemplary safety management</li>
                <li><strong>11 studies >90 DQI</strong> - strong overall performance</li>
                <li><strong>85.4% low-risk</strong> patients across portfolio</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ROI Calculator
    st.subheader("💰 Interactive ROI Calculator")
    st.markdown("**Model the business impact of implementing recommended improvements**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Current State")
        current_dqi = st.slider("Current Average DQI", 0, 100, 74, key="current")
        current_cras = st.number_input("Number of CRAs", 1, 20, 10, key="cras")
        weeks_to_lock = st.number_input("Weeks to Database Lock", 1, 20, 8, key="weeks")
    
    with col2:
        st.markdown("##### Proposed Changes")
        add_cras = st.slider("Additional CRAs to Hire", 0, 5, 2)
        automate_sae = st.checkbox("Implement Automated SAE Alerts", value=True)
        increase_visits = st.checkbox("Increase Site Visit Frequency", value=True)
    
    # Calculate impact
    dqi_improvement = 0
    if add_cras > 0:
        dqi_improvement += add_cras * 5  # 5 points per CRA
    if automate_sae:
        dqi_improvement += 10  # 10 points for automation
    if increase_visits:
        dqi_improvement += 5  # 5 points for more visits
    
    new_dqi = min(current_dqi + dqi_improvement, 100)
    time_saved = int(dqi_improvement * 0.4)  # 0.4 weeks saved per DQI point
    new_timeline = max(weeks_to_lock - time_saved, 2)
    
    cost_per_cra = 20000
    automation_cost = 30000
    visit_cost = 10000
    total_cost = (add_cras * cost_per_cra) + (automation_cost if automate_sae else 0) + (visit_cost if increase_visits else 0)
    
    revenue_per_week = 350000  # Revenue impact of faster launch
    revenue_gained = time_saved * revenue_per_week
    net_roi = revenue_gained - total_cost
    
    st.markdown("---")
    st.subheader("📊 Projected Outcomes")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("New DQI", f"{new_dqi:.0f}", f"+{dqi_improvement:.0f} points")
    with col2:
        st.metric("New Timeline", f"{new_timeline:.0f} weeks", f"-{time_saved:.0f} weeks")
    with col3:
        st.metric("Implementation Cost", f"${total_cost:,}")
    with col4:
        st.metric("NET ROI", f"${net_roi:,}", 
                 f"{net_roi/total_cost*100:.0f}% return" if total_cost > 0 else "N/A")
    
    # Visualization
    comparison_data = pd.DataFrame({
        'Scenario': ['Current', 'With Improvements'],
        'DQI': [current_dqi, new_dqi],
        'Timeline (weeks)': [weeks_to_lock, new_timeline],
        'Cost ($000s)': [0, total_cost/1000]
    })
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('DQI Improvement', 'Timeline Reduction')
    )
    
    fig.add_trace(
        go.Bar(x=comparison_data['Scenario'], y=comparison_data['DQI'], 
               marker_color=['#dc3545', '#28a745'], name='DQI'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=comparison_data['Scenario'], y=comparison_data['Timeline (weeks)'],
               marker_color=['#dc3545', '#28a745'], name='Timeline'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations summary
    if net_roi > 0:
        st.markdown(f"""
        <div class="success-box">
            <h4>✅ RECOMMENDATION: Proceed with Implementation</h4>
            <p><strong>Expected ROI:</strong> ${net_roi:,} ({net_roi/total_cost*100:.0f}% return)</p>
            <p><strong>Time to Market:</strong> {time_saved:.0f} weeks faster</p>
            <p><strong>Risk Mitigation:</strong> Improved regulatory compliance</p>
            <p><strong>Implementation Time:</strong> 4-6 weeks</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Consider adjusting parameters for better ROI</h4>
            <p>Current configuration may not provide sufficient return on investment.</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# GEN-AI MODEL SETUP
# ============================================================================
import os
HF_TOKEN = os.getenv("HF_TOKEN")

# ============================================================================
# GEN-AI MODEL SETUP
# ============================================================================

def call_genai(prompt):
    import json

    url = "https://router.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": """
You are an AI assistant for CLINICAL TRIAL DATA QUALITY only.

RULES:
- Do NOT perform numeric calculations yourself.
- Do NOT guess or estimate numbers.
- Do NOT invent formulas.
- ONLY use numbers provided by the system.
- If numbers are not supplied, reply:
  "I cannot calculate values, but I can explain insights."
- Never give medical or treatment advice.
- Focus on DQI, anomalies, risk, SAE, overdue visits, trends and explanations.
"""
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.4
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code != 200:
        return f"HTTP {response.status_code}: {response.text}"

    result = response.json()

    try:
        return result["choices"][0]["message"]["content"]
    except Exception:
        return f"Unexpected response format: {result}"


# ============================================================================
# PYTHON BACKEND — REAL DATA ANSWERS (NO HALLUCINATION)
# ============================================================================

def answer_with_real_data(user_question):
    q = user_question.lower()

    # ---- average DQI of specific study ----
    if "average dqi" in q or "avg dqi" in q:
        for study in df["Study"].unique():
            if study.lower() in q:
                value = df[df["Study"] == study]["Smart_DQI_Noisy"].mean()
                return f"The actual average DQI for {study} is {value:.2f}."

    # ---- total high-risk patients ----
    if "high risk" in q and "patient" in q:
        count = (df["Smart_Risk_Noisy"] == "High").sum()
        return f"There are {count} high-risk patients in the dataset."

    # ---- anomalies detected ----
    if "anomal" in q:
        count = anomalies["Is_Anomaly"].sum()
        return f"A total of {count} anomalies were detected by the ML model."

    # ---- total SAE pending ----
    if "sae" in q and "pending" in q:
        value = df["SAE_Pending_Count"].sum()
        return f"There are {value} SAE pending events in total across all patients."

    # ----------------------------------------------------------------
    # ⭐ WHY IS STUDY / SITE HIGH RISK — REAL REASONS
    # ----------------------------------------------------------------

    if "why" in q and "high risk" in q:

        # ---- study-level explanation ----
        for study in df["Study"].unique():
            if study.lower() in q:
                sub = df[df["Study"] == study]

                avg_dqi = sub["Smart_DQI_Noisy"].mean()
                high_risk_pct = (sub["Smart_Risk_Noisy"] == "High").mean() * 100
                sae_total = sub["SAE_Pending_Count"].sum()
                overdue_total = sub["Overdue_Visits_Count"].sum()

                anomaly_total = anomalies[anomalies["Study"] == study]["Is_Anomaly"].sum()

                return (
                    f"{study} is considered high risk due to:\n"
                    f"- Average DQI = {avg_dqi:.2f}\n"
                    f"- High-risk patients = {high_risk_pct:.1f}%\n"
                    f"- Total SAE pending = {sae_total}\n"
                    f"- Total overdue visits = {overdue_total}\n"
                    f"- Anomalies detected = {anomaly_total}"
                )

        # ---- site-level explanation ----
        if "site" in q:
            for site in df["Site ID"].unique():
                if str(site).lower() in q:
                    sub = df[df["Site ID"] == site]

                    avg_dqi = sub["Smart_DQI_Noisy"].mean()
                    high_risk_pct = (sub["Smart_Risk_Noisy"] == "High").mean() * 100
                    sae_total = sub["SAE_Pending_Count"].sum()
                    overdue_total = sub["Overdue_Visits_Count"].sum()

                    return (
                        f"Site {site} shows high risk indicators:\n"
                        f"- Average DQI = {avg_dqi:.2f}\n"
                        f"- High-risk patients = {high_risk_pct:.1f}%\n"
                        f"- Total SAE pending = {sae_total}\n"
                        f"- Total overdue visits = {overdue_total}"
                    )

    return None



# ============================================================================
# 📄 AI REPORT TEXT BUILDER (uses real dataset numbers)
# ============================================================================

def build_ai_report_text():

    avg_dqi = df["Smart_DQI_Noisy"].mean()
    high_risk = (df["Smart_Risk_Noisy"] == "High").sum()
    anomaly_count = anomalies["Is_Anomaly"].sum()

    prompt = f"""
Write an executive clinical data quality report.

Include:
- overall data quality
- risk levels
- SAE and overdue visit concerns
- anomaly burden
- recommended data quality actions

ONLY use the numbers below exactly:
Average DQI = {avg_dqi:.2f}
High-risk patients = {high_risk}
Anomalies detected = {anomaly_count}

Do NOT invent numbers.
Do NOT give medical guidance.
Bullet points preferred.
"""

    return call_genai(prompt)

def build_study_report_text(study, language):

    sub = df[df["Study"] == study]

    avg_dqi = sub["Smart_DQI_Noisy"].mean()
    high_risk = (sub["Smart_Risk_Noisy"] == "High").sum()
    sae = sub["SAE_Pending_Count"].sum()
    overdue = sub["Overdue_Visits_Count"].sum()

    prompt = f"""
Write a clinical DATA QUALITY report for study: {study}

Use ONLY these exact numbers:
Average DQI = {avg_dqi:.2f}
High-risk patients = {high_risk}
SAE pending = {sae}
Overdue visits = {overdue}

Focus only on:
- data quality
- risk
- anomalies
- monitoring needs

Do NOT give medical advice.
Write bullet points.

Language required: {language}
"""

    return call_genai(prompt)

# ============================================================================
# 📄 PDF GENERATOR
# ============================================================================

def generate_pdf_report(ai_text):
    buffer = io.BytesIO()

    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y, "Clinical Trial Data Quality AI Report")
    y -= 30

    p.setFont("Helvetica", 10)

    for line in ai_text.split("\n"):
        if y < 50:
            p.showPage()
            y = height - 50
            p.setFont("Helvetica", 10)
        p.drawString(50, y, line[:95])
        y -= 15

    p.save()
    buffer.seek(0)
    return buffer

import tempfile
from plotly.io import write_image

def create_study_graphs(study):

    sub = df[df["Study"] == study]

    # Risk distribution
    fig1 = px.histogram(sub, x="Smart_DQI_Noisy", nbins=20, title="DQI Distribution")

    # SAE vs Overdue
    fig2 = px.scatter(
        sub,
        x="SAE_Pending_Count",
        y="Overdue_Visits_Count",
        title="SAE vs Overdue Visits"
    )

    temp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")

    write_image(fig1, temp1.name)
    write_image(fig2, temp2.name)

    return temp1.name, temp2.name

# ============================================================================
# PAGE 7: Gen-AI Assistant
# ============================================================================
if page == "🧠 Gen-AI Assistant":

    st.header("🧠 Gen-AI Assistant for Clinical Trial Data Quality")

    st.markdown("""
Ask natural-language questions about:
• DQI and risk
• anomalies
• overdue visits
• SAE pending
• why a STUDY or SITE is high-risk

📊 Python computes real numbers  
🤖 AI explains in natural language  
🔒 No medical advice  
""")

    user_question = st.text_area("💬 Type your question:")

    if st.button("Generate Answer"):
        if user_question.strip() == "":
            st.warning("Please enter a question first.")
        else:
            with st.spinner("AI is thinking..."):

                numeric_answer = answer_with_real_data(user_question)

                if numeric_answer:
                    prompt = f"""
Here is the correct numeric answer from Python:
{numeric_answer}

Explain clearly in 4–6 bullet points.
Do NOT change the numbers.
"""
                    answer = call_genai(prompt)
                else:
                    context = df.head(40).to_string()
                    prompt = f"""
Use dataset context to answer.
Do NOT invent numbers.

{context}

Question:
{user_question}
"""
                    answer = call_genai(prompt)

                st.success(answer)


def generate_study_pdf(study, ai_text, img1, img2):

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)

    width, height = letter
    y = height - 40

    p.setFont("Helvetica-Bold", 14)
    p.drawString(40, y, f"Study Report: {study}")
    y -= 30

    p.setFont("Helvetica", 10)

    for line in ai_text.split("\n"):
        if y < 120:
            p.showPage()
            y = height - 40
            p.setFont("Helvetica", 10)
        p.drawString(40, y, line[:95])
        y -= 14

    # Insert charts
    p.showPage()
    p.drawImage(img1, 40, 300, width=520, height=350)
    p.drawImage(img2, 40, 20, width=520, height=260)

    p.save()
    buffer.seek(0)
    return buffer


# # ========================================================================
# # PAGE 8 :Early Warning Predictor
# # ========================================================================
# df, anomalies, study_comparison = load_data()
# # ---------- SAFELY CREATE Total_Issues COLUMN ----------
# if "Total_Issues" not in df.columns:
#     df["Total_Issues"] = (
#         df["SAE_Pending_Count"]
#         + df["Overdue_Visits_Count"]
#         + df["Missing_Pages"]
#     )

# early_model, early_features = joblib.load("early_warning_model.pkl")
# if page == "⚠️ Early Warning Predictor":

#     st.header("⚠️ Early-Warning High-Risk Prediction System")
#     st.markdown("Predict patients **likely to become High-Risk soon**")

#     # prepare features
#     # ensure all expected features exist
#     for c in early_features:
#        if c not in df.columns:
#           df[c] = 0

#     X_new = df[early_features]

#     risk_prob = early_model.predict_proba(X_new)[:, 1]

#     df["Future_HighRisk_Probability"] = (risk_prob * 100).round(1)

#     # classify buckets
#     def bucket(p):
#         if p >= 70:
#             return "🚨 Very Likely High-Risk"
#         elif p >= 40:
#             return "⚠️ At Risk"
#         else:
#             return "🟢 Stable"

#     df["Early_Warning_Label"] = df["Future_HighRisk_Probability"].apply(bucket)

#     st.subheader("📊 Prediction Summary")

#     c1, c2, c3 = st.columns(3)
#     with c1:
#         st.metric("🚨 Very Likely High-Risk", (df["Early_Warning_Label"] == "🚨 Very Likely High-Risk").sum())
#     with c2:
#         st.metric("⚠️ At Risk", (df["Early_Warning_Label"] == "⚠️ At Risk").sum())
#     with c3:
#         st.metric("🟢 Stable", (df["Early_Warning_Label"] == "🟢 Stable").sum())

#     st.markdown("---")

#     st.subheader("📋 Patient-Level Predictions")

#     display = df[[
#         "Subject ID",
#         "Study",
#         "Smart_DQI_Noisy",
#         "SAE_Pending_Count",
#         "Overdue_Visits_Count",
#         "Missing_Pages",
#         "Future_HighRisk_Probability",
#         "Early_Warning_Label"
#     ]].sort_values("Future_HighRisk_Probability", ascending=False)

#     st.dataframe(display, use_container_width=True, height=500)

# st.subheader("🧠 Explain Prediction")

# pid = st.selectbox(
#     "Select patient:",
#     display["Subject ID"].astype(str).tolist()
# )

# if st.button("Explain risk score"):
#     row = display[display["Subject ID"].astype(str) == pid].iloc[0]

#     prompt = f"""
# Explain why this patient is at future high-risk.

# Use ONLY the numbers given.

# Patient: {pid}
# Probability becoming high-risk: {row['Future_HighRisk_Probability']}%

# Current Data:
# DQI = {row['Smart_DQI_Noisy']}
# SAE pending = {row['SAE_Pending_Count']}
# Overdue visits = {row['Overdue_Visits_Count']}
# Missing pages = {row['Missing_Pages']}

# Do NOT do math.
# Do NOT change numbers.
# Focus on data quality, monitoring risk, and operational reasons.
# """

#     exp = call_genai(prompt)
#     st.success(exp)



# ========================================================================
# PDF REPORT BUTTON
# ========================================================================

st.markdown("## 📄 Per-Study AI PDF Report")

study_choice = st.selectbox("Select study:", sorted(df["Study"].unique()))

language = st.selectbox(
    "Select language:",
    ["English", "Hindi", "French", "German"]
)

if st.button("Generate Study PDF"):
    with st.spinner("Building study report..."):

        ai_text = build_study_report_text(study_choice, language)

        img1, img2 = create_study_graphs(study_choice)

        pdf_buffer = generate_study_pdf(study_choice, ai_text, img1, img2)

        st.success("Study PDF ready!")

        st.download_button(
            label="⬇️ Download Study Report PDF",
            data=pdf_buffer,
            file_name=f"{study_choice}_Report.pdf",
            mime="application/pdf"
        )
