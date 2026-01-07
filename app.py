import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Clinical Trial Data Quality Dashboard", layout="wide", page_icon="🏥")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('outputs/all_studies_smart_dqi.csv')

df = load_data()

# Title
st.title("🏥 Clinical Trial Data Quality Intelligence Dashboard")
st.markdown("### Context-Aware Risk Assessment Across Multiple Studies")

# Sidebar Filters
st.sidebar.header("🔍 Filters")

# Select Studies
selected_studies = st.sidebar.multiselect(
    "Select Studies",
    options=df['Study'].unique(),
    default=df['Study'].unique()
)

# Select Contexts
selected_contexts = st.sidebar.multiselect(
    "Select Contexts",
    options=df['Context'].unique(),
    default=df['Context'].unique()
)

# Select Risk Levels
selected_risks = st.sidebar.multiselect(
    "Risk Level",
    options=['High', 'Medium', 'Low'],
    default=['High', 'Medium', 'Low']
)

# Apply filters
df_filtered = df[
    (df['Study'].isin(selected_studies)) &
    (df['Context'].isin(selected_contexts)) &
    (df['Smart_Risk'].isin(selected_risks))
]

# SAFETY CHECK: Prevent division by zero
if len(df_filtered) == 0:
    st.error("⚠️ No data matches the selected filters. Please adjust your filter selection.")
    st.stop()

# KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Patients", f"{len(df_filtered):,}")

with col2:
    avg_dqi = df_filtered['Smart_DQI'].mean()
    st.metric("Avg Smart DQI", f"{avg_dqi:.1f}")

with col3:
    high_risk = len(df_filtered[df_filtered['Smart_Risk'] == 'High'])
    high_risk_pct = (high_risk / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
    st.metric("High Risk Patients", high_risk, delta=f"{high_risk_pct:.1f}%", delta_color="inverse")

with col4:
    total_sae = int(df_filtered['SAE_Pending_Count'].sum())
    st.metric("Total Pending SAE", total_sae)

st.markdown("---")

# Row 1: Risk Distribution + Smart DQI by Study
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Risk Distribution")
    risk_counts = df_filtered['Smart_Risk'].value_counts()
    
    if len(risk_counts) > 0:
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
            hole=0.4
        )
        fig_risk.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.info("No risk data to display")

with col2:
    st.subheader("📈 Smart DQI by Study")
    study_dqi = df_filtered.groupby('Study')['Smart_DQI'].mean().reset_index()
    study_dqi = study_dqi.sort_values('Smart_DQI')
    
    if len(study_dqi) > 0:
        fig_dqi = px.bar(
            study_dqi,
            x='Smart_DQI',
            y='Study',
            orientation='h',
            color='Smart_DQI',
            color_continuous_scale=['red', 'yellow', 'green'],
            range_color=[0, 100]
        )
        fig_dqi.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_dqi, use_container_width=True)
    else:
        st.info("No study data to display")

st.markdown("---")

# Row 2: Context Comparison + Feature Impact
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Context Comparison")
    context_data = df_filtered.groupby('Context').agg({
        'Subject ID': 'count',
        'Smart_DQI': 'mean'
    }).reset_index()
    context_data.columns = ['Context', 'Patients', 'Avg Smart DQI']
    
    if len(context_data) > 0:
        fig_context = px.bar(
            context_data,
            x='Context',
            y='Avg Smart DQI',
            color='Patients',
            color_continuous_scale='Blues',
            text='Avg Smart DQI'
        )
        fig_context.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_context.update_layout(height=400)
        st.plotly_chart(fig_context, use_container_width=True)
    else:
        st.info("No context data to display")

with col2:
    st.subheader("🔬 Feature Impact on Risk")
    
    # Calculate correlations with Smart_Risk (convert to numeric)
    df_filtered_copy = df_filtered.copy()
    risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df_filtered_copy['Risk_Numeric'] = df_filtered_copy['Smart_Risk'].map(risk_map)
    
    correlations = {
        'SAE Pending': df_filtered_copy['SAE_Pending_Count'].corr(df_filtered_copy['Risk_Numeric']),
        'Overdue Visits': df_filtered_copy['Overdue_Visits_Count'].corr(df_filtered_copy['Risk_Numeric']),
        'Missing Pages': df_filtered_copy['Missing_Pages'].corr(df_filtered_copy['Risk_Numeric'])
    }
    
    # Remove NaN correlations
    correlations = {k: v for k, v in correlations.items() if pd.notna(v)}
    
    if len(correlations) > 0:
        fig_corr = go.Figure(go.Bar(
            x=list(correlations.values()),
            y=list(correlations.keys()),
            orientation='h',
            marker=dict(color=list(correlations.values()), 
                       colorscale='RdYlGn_r',
                       cmin=-1, cmax=1)
        ))
        fig_corr.update_layout(
            xaxis_title="Correlation with Risk",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Insufficient data for correlation analysis")

st.markdown("---")

# Row 3: High-Risk Patients Details
st.subheader("🚨 High-Risk Patients Details")

high_risk_patients = df_filtered[df_filtered['Smart_Risk'] == 'High']

if len(high_risk_patients) > 0:
    high_risk_display = high_risk_patients.nsmallest(10, 'Smart_DQI')[[
        'Subject ID', 'Site ID', 'Study', 'Smart_DQI', 
        'SAE_Pending_Count', 'Overdue_Visits_Count', 'Missing_Pages'
    ]]
    st.dataframe(high_risk_display, use_container_width=True, hide_index=True)
else:
    st.success("✅ No high-risk patients in selected filters!")

st.markdown("---")

# Row 4: Smart DQI vs Basic DQI Comparison
st.subheader("⚖️ Smart DQI vs Basic DQI Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### DQI Scatter Plot")
    
    if len(df_filtered) > 0:
        fig_scatter = px.scatter(
            df_filtered,
            x='Basic_DQI',
            y='Smart_DQI',
            color='Smart_Risk',
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
            opacity=0.6,
            hover_data=['Subject ID', 'Study']
        )
        fig_scatter.add_shape(
            type="line",
            x0=0, y0=0, x1=100, y1=100,
            line=dict(color="gray", dash="dash")
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No data to display")

with col2:
    st.markdown("#### Distribution of DQI Differences")
    
    if len(df_filtered) > 0:
        fig_diff = px.histogram(
            df_filtered,
            x='Difference',
            nbins=50,
            color_discrete_sequence=['steelblue']
        )
        fig_diff.update_layout(
            xaxis_title="Smart DQI - Basic DQI",
            yaxis_title="Count",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_diff, use_container_width=True)
    else:
        st.info("No data to display")

st.markdown("---")

# Key Insights
st.subheader("💡 Key Insights")

if len(df_filtered) > 0:
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("#### Smart DQI Analysis Summary:")
        st.write(f"- **Total Patients Analyzed:** {len(df_filtered):,}")
        st.write(f"- **Studies Included:** {', '.join(selected_studies)}")
        st.write(f"- **High-Risk Patients:** {len(high_risk_patients)} ({len(high_risk_patients)/len(df_filtered)*100:.1f}%)")
        st.write(f"- **Primary Risk Driver:** SAE Pending Reviews ({df_filtered['SAE_Pending_Count'].sum():.0f} total)")
        st.write(f"- **Context-Aware Weighting:** {'Oncology trials prioritize safety (32% weight)' if 'oncology' in str(selected_contexts).lower() else 'Standard weighting applied'}")
    
    with insights_col2:
        st.markdown("#### Actionable Recommendations:")
        
        # Dynamic recommendations based on data
        if len(high_risk_patients) > 0:
            worst_study = high_risk_patients.groupby('Study').size().idxmax()
            st.write(f"1. **Urgent:** Focus on {worst_study} - highest number of high-risk patients")
        
        total_sae = df_filtered['SAE_Pending_Count'].sum()
        if total_sae > 100:
            st.write(f"2. **Critical:** {total_sae:.0f} pending SAE reviews require immediate attention")
        elif total_sae > 0:
            st.write(f"2. **Action:** {total_sae:.0f} pending SAE reviews need resolution")
        else:
            st.write("2. ✅ All SAE reviews completed - excellent safety compliance")
        
        overdue_visits = df_filtered['Overdue_Visits_Count'].sum()
        if overdue_visits > 50:
            st.write(f"3. **Follow-up:** {overdue_visits:.0f} overdue visits need scheduling")
        elif overdue_visits > 0:
            st.write(f"3. **Monitor:** {overdue_visits:.0f} overdue visits to track")
        else:
            st.write("3. ✅ Visit schedules on track")
        
        clean_patients = len(df_filtered[df_filtered['Smart_Risk'] == 'Low'])
        clean_pct = (clean_patients / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        if clean_pct >= 70:
            st.write(f"4. ✅ Study quality excellent: {clean_pct:.1f}% low-risk patients")
        elif clean_pct >= 50:
            st.write(f"4. ⚠️ Study quality acceptable: {clean_pct:.1f}% low-risk patients")
        else:
            st.write(f"4. 🚨 Study quality needs improvement: only {clean_pct:.1f}% low-risk patients")
else:
    st.info("Select filters to view insights")

# Footer
st.markdown("---")
st.caption("💡 Smart DQI uses context-aware weights based on therapeutic area, trial phase, and timeline to provide more accurate risk assessment than traditional DQI scoring.")
