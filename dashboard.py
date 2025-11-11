import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime


# Page configuration
st.set_page_config(
    page_title="Pipeline Corrosion Forecasting Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional sidebar
st.markdown("""
<style>
    .nav-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .nav-title {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Load your data
@st.cache_data
def load_data():
    return pd.read_csv('pipeline_predictions.csv')


df = load_data()


# Apply filters function
def apply_filters(data, material_filter, risk_filter, pressure_range,
                  corrosion_rate_filter, age_range, thickness_loss_range,
                  pipe_size_range, prediction_confidence):
    filtered_data = data.copy()

    if material_filter:
        filtered_data = filtered_data[filtered_data['material'].isin(material_filter)]

    if risk_filter:
        filtered_data = filtered_data[filtered_data['Risk_Level'].isin(risk_filter)]

    filtered_data = filtered_data[
        (filtered_data['max_pressure_psi'] >= pressure_range[0]) &
        (filtered_data['max_pressure_psi'] <= pressure_range[1])
        ]

    filtered_data = filtered_data[
        (filtered_data['corrosion_rate_mm_per_year'] >= corrosion_rate_filter[0]) &
        (filtered_data['corrosion_rate_mm_per_year'] <= corrosion_rate_filter[1])
        ]

    filtered_data = filtered_data[
        (filtered_data['time_years'] >= age_range[0]) &
        (filtered_data['time_years'] <= age_range[1])
        ]

    filtered_data = filtered_data[
        (filtered_data['thickness_loss_mm'] >= thickness_loss_range[0]) &
        (filtered_data['thickness_loss_mm'] <= thickness_loss_range[1])
        ]

    filtered_data = filtered_data[
        (filtered_data['pipe_size_mm'] >= pipe_size_range[0]) &
        (filtered_data['pipe_size_mm'] <= pipe_size_range[1])
        ]

    filtered_data = filtered_data[
        filtered_data['Prediction_Confidence'] >= prediction_confidence
        ]

    return filtered_data


# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "monitoring"

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div class="nav-container">
        <div class="nav-title">Pipeline Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation buttons
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        monitoring_active = st.session_state.current_page == "monitoring"
        if st.button("Monitor",
                     use_container_width=True,
                     type="primary" if monitoring_active else "secondary"):
            st.session_state.current_page = "monitoring"

    with col2:
        simulator_active = st.session_state.current_page == "simulator"
        if st.button("Simulation",
                     use_container_width=True,
                     type="primary" if simulator_active else "secondary"):
            st.session_state.current_page = "simulator"

    with col3:
        batch_active = st.session_state.current_page == "batch"
        if st.button("Bulk Analysis",
                     use_container_width=True,
                     type="primary" if batch_active else "secondary"):
            st.session_state.current_page = "batch"

    with col4:
        models_active = st.session_state.current_page == "models"
        if st.button("Models",
                     use_container_width=True,
                     type="primary" if models_active else "secondary"):
            st.session_state.current_page = "models"

    st.markdown("---")

    # Page descriptions
    if st.session_state.current_page == "monitoring":
        st.subheader("Monitoring Dashboard")
        st.caption("Real-time pipeline corrosion monitoring with interactive filters and alerts")

    elif st.session_state.current_page == "simulator":
        st.subheader("Corrosion Simulator")
        st.caption("Predict future corrosion scenarios and simulate maintenance strategies")

    elif st.session_state.current_page == "batch":
        st.subheader("Bulk Pipeline Analysis")
        st.caption("Upload CSV files for comprehensive pipeline fleet risk assessment")

    elif st.session_state.current_page == "models":
        st.subheader("ML Model Performance")
        st.caption("Evaluate and compare machine learning model performance")

# Map to page names
page_mapping = {
    "monitoring": "Monitoring Dashboard",
    "simulator": "Corrosion Simulator",
    "batch": "Bulk Pipeline Analysis",
    "models": "ML Model Performance"
}

page = page_mapping[st.session_state.current_page]


# PAGE 1: MONITORING DASHBOARD
if page == "Monitoring Dashboard":
    st.title("Pipeline Monitoring Dashboard")
    st.markdown("Real-time monitoring and analysis of pipeline corrosion data")

    # Display dataset info
    st.sidebar.markdown(f"**Dataset Info:** {len(df)} total pipelines")

    # Filters in sidebar for monitoring page only
    with st.sidebar:
        st.header("Monitoring Filters")

        # Critical Engineering Filters
        st.subheader("Pipeline Characteristics")

        # Get unique values from your actual data
        available_materials = df['material'].unique().tolist() if 'material' in df.columns else []
        available_risk_levels = df['Risk_Level'].unique().tolist() if 'Risk_Level' in df.columns else []

        material_filter = st.multiselect(
            "Pipeline Material",
            available_materials,
            default=available_materials
        )

        risk_filter = st.multiselect(
            "Risk Level",
            available_risk_levels,
            default=available_risk_levels
        )

        # Operational Parameters
        st.subheader("Operational Parameters")

        # Get min/max values from your actual data
        pressure_min = df['max_pressure_psi'].min() if 'max_pressure_psi' in df.columns else 0
        pressure_max = df['max_pressure_psi'].max() if 'max_pressure_psi' in df.columns else 500

        pressure_range = st.slider(
            "Operating Pressure (PSI)",
            min_value=int(pressure_min),
            max_value=int(pressure_max),
            value=(int(pressure_min), int(pressure_max))
        )

        corrosion_min = df['corrosion_rate_mm_per_year'].min() if 'corrosion_rate_mm_per_year' in df.columns else 0.0
        corrosion_max = df['corrosion_rate_mm_per_year'].max() if 'corrosion_rate_mm_per_year' in df.columns else 2.0

        corrosion_rate_filter = st.slider(
            "Corrosion Rate (mm/year)",
            min_value=float(corrosion_min),
            max_value=float(corrosion_max),
            value=(float(corrosion_min), float(corrosion_max))
        )

        # Time-based Filters
        st.subheader("Age & Condition")

        age_min = df['time_years'].min() if 'time_years' in df.columns else 0
        age_max = df['time_years'].max() if 'time_years' in df.columns else 50

        age_range = st.slider(
            "Pipeline Age (years)",
            min_value=int(age_min),
            max_value=int(age_max),
            value=(int(age_min), int(age_max))
        )

        thickness_min = df['thickness_loss_mm'].min() if 'thickness_loss_mm' in df.columns else 0.0
        thickness_max = df['thickness_loss_mm'].max() if 'thickness_loss_mm' in df.columns else 10.0

        thickness_loss_range = st.slider(
            "Thickness Loss (mm)",
            min_value=float(thickness_min),
            max_value=float(thickness_max),
            value=(float(thickness_min), float(thickness_max))
        )

        # Additional filters - ADD THE MISSING VARIABLES
        pipe_size_min = df['pipe_size_mm'].min() if 'pipe_size_mm' in df.columns else 0
        pipe_size_max = df['pipe_size_mm'].max() if 'pipe_size_mm' in df.columns else 1000

        pipe_size_range = st.slider(
            "Pipe Size (mm)",
            min_value=int(pipe_size_min),
            max_value=int(pipe_size_max),
            value=(int(pipe_size_min), int(pipe_size_max))
        )

        confidence_min = df['Prediction_Confidence'].min() if 'Prediction_Confidence' in df.columns else 0.0
        confidence_max = df['Prediction_Confidence'].max() if 'Prediction_Confidence' in df.columns else 1.0

        prediction_confidence = st.slider(
            "Minimum Prediction Confidence",
            min_value=float(confidence_min),
            max_value=float(confidence_max),
            value=0.7
        )

    # Apply filters - NOW ALL VARIABLES ARE DEFINED
    filtered_data = apply_filters(
        df, material_filter, risk_filter, pressure_range,
        corrosion_rate_filter, age_range, thickness_loss_range,
        pipe_size_range, prediction_confidence
    )

    # CLEAN EXECUTIVE SUMMARY LIKE THE REFERENCE IMAGE
    st.header("Pipeline Performance Overview")

    # Calculate key metrics
    total_pipes = len(filtered_data)
    critical_pipes = filtered_data[
        filtered_data['Risk_Level'] == 'Critical'] if 'Risk_Level' in filtered_data.columns else pd.DataFrame()
    high_risk_pipes = filtered_data[
        filtered_data['Risk_Level'] == 'High'] if 'Risk_Level' in filtered_data.columns else pd.DataFrame()

    critical_count = len(critical_pipes)
    high_risk_count = len(high_risk_pipes)
    total_at_risk = critical_count + high_risk_count

    # Calculate comparison with previous period (simulated)
    previous_period_risk = total_at_risk * 0.8  # Simulating 20% improvement
    risk_change = total_at_risk - previous_period_risk
    risk_change_percent = (risk_change / previous_period_risk * 100) if previous_period_risk > 0 else 0

    # Row 1: Key Performance Indicators - ALL DYNAMIC & ACTIONABLE
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # HIGH RISK PIPES (replaces Critical Risk)
        if 'Risk_Level' in filtered_data.columns:
            high_risk_count = len(filtered_data[
                                      filtered_data['Risk_Level'].isin(['High Risk', 'Critical', 'High'])
                                  ])
        else:
            high_risk_count = 0

        st.metric(
            label="🚨 High Risk Pipes",
            value=high_risk_count
        )
        st.caption("Pipes needing immediate attention")

    with col2:
        # AVG CORROSION RATE
        if 'corrosion_rate_mm_per_year' in filtered_data.columns and not filtered_data.empty:
            current_corrosion = filtered_data['corrosion_rate_mm_per_year'].mean()
            st.metric(
                label="🔥 Avg Corrosion Rate",
                value=f"{current_corrosion:.2f} mm/yr"
            )
            st.caption("Industry avg: <0.3 mm/yr")
        else:
            st.metric(label="🔥 Avg Corrosion Rate", value="N/A")

    with col3:
        # PREDICTION CONFIDENCE (keep this - it's good!)
        if 'Prediction_Confidence' in filtered_data.columns and not filtered_data.empty:
            avg_confidence = filtered_data['Prediction_Confidence'].mean()
            st.metric(
                label="🎯 Prediction Confidence",
                value=f"{(avg_confidence * 100):.1f}%"
            )
            st.caption("Model reliability")
        else:
            st.metric(label="🎯 Prediction Confidence", value="N/A")

    # with col4:
    #     # URGENT MAINTENANCE (keep but fix the logic)
    #     if 'Maintenance_Recommendation' in filtered_data.columns and not filtered_data.empty:
    #         urgent_count = len(filtered_data[
    #                                filtered_data['Maintenance_Recommendation'].str.contains(
    #                                    'IMMEDIATE|URGENT', case=False, na=False
    #                                )
    #                            ])
    #         st.metric(
    #             label="🔧 Urgent Maintenance",
    #             value=urgent_count
    #         )
    #         st.caption("Schedule inspections now")
    #     else:
    #         st.metric(label="🔧 Urgent Maintenance", value="N/A")

    # Row 2: NEW DYNAMIC METRICS
    st.markdown("---")
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        # CORROSION HOTSPOTS (new - identifies problem areas)
        if 'corrosion_rate_mm_per_year' in filtered_data.columns and not filtered_data.empty:
            # Count pipes with dangerously high corrosion (>1.0 mm/year)
            high_corrosion_count = len(filtered_data[
                                           filtered_data['corrosion_rate_mm_per_year'] > 1.0
                                           ])
            st.metric(
                label="🔥 Corrosion Hotspots",
                value=high_corrosion_count
            )
            st.caption("Corrosion > 1.0 mm/year")
        else:
            st.metric(label="🔥 Corrosion Hotspots", value="N/A")

    with col6:
        # TIME TO FAILURE ESTIMATE (new - very engineering!)
        # WATER PIPELINE SPECIFIC REMAINING LIFE WITH SIZE-BASED MINIMUM
        if 'corrosion_rate_mm_per_year' in filtered_data.columns and 'thickness_mm' in filtered_data.columns and 'pipe_size_mm' in filtered_data.columns:
            if not filtered_data.empty:
                design_life = 50  # years - typical water infrastructure

                life_data = filtered_data[['thickness_mm', 'corrosion_rate_mm_per_year', 'pipe_size_mm']].copy()


                # Calculate minimum required thickness based on pipe size
                def get_min_thickness(size):
                    if size <= 100:
                        return 3.0
                    elif size <= 300:
                        return 4.0
                    elif size <= 600:
                        return 6.0
                    elif size <= 900:
                        return 8.0
                    else:
                        return 10.0


                life_data['min_required_thickness'] = life_data['pipe_size_mm'].apply(get_min_thickness)

                # Set realistic minimum corrosion rate (your data shows 0.0005 is too low)
                life_data['effective_corrosion'] = life_data['corrosion_rate_mm_per_year'].clip(lower=0.05)
                # ^^^ 0.05 mm/year is realistic minimum for water pipelines

                # Calculate remaining life: (current - minimum) / corrosion rate
                life_data['remaining_thickness'] = life_data['thickness_mm'] - life_data['min_required_thickness']
                life_data['remaining_thickness'] = life_data['remaining_thickness'].clip(lower=0)

                life_data['remaining_life'] = life_data['remaining_thickness'] / life_data['effective_corrosion']

                # Cap at realistic maximum
                life_data['remaining_life'] = life_data['remaining_life'].clip(upper=80)

                avg_remaining_life = life_data['remaining_life'].median()

                st.metric(
                    label="⏰ Remaining Life",
                    value=f"{avg_remaining_life:.1f} yrs"
                )
                st.caption("Size-based minimum thickness (3-10mm)")
            else:
                st.metric(label="⏰ Remaining Life", value="N/A")
        else:
            st.metric(label="⏰ Remaining Life", value="N/A")

    with col7:
        # SAFETY MARGIN (new - engineering critical!)
        if 'thickness_mm' in filtered_data.columns and not filtered_data.empty:
            # Assume minimum safe thickness is 5mm
            min_safe_thickness = 5.0
            pipes_below_safe = len(filtered_data[filtered_data['thickness_mm'] < min_safe_thickness])
            st.metric(
                label="🛡️ Below Safety Margin",
                value=pipes_below_safe
            )
            st.caption(f"Thickness < {min_safe_thickness}mm")
        else:
            st.metric(label="🛡️ Below Safety Margin", value="N/A")

    # with col8:
    #     # INSPECTION BACKLOG (new - operational!)
    #     if 'Maintenance_Recommendation' in filtered_data.columns and not filtered_data.empty:
    #         maintenance_backlog = len(filtered_data[
    #                                       ~filtered_data['Maintenance_Recommendation'].str.contains('6 months',
    #                                                                                                 case=False,
    #                                                                                                 na=False)
    #                                   ])
    #         st.metric(
    #             label="📋 Maintenance Backlog",
    #             value=maintenance_backlog
    #         )
    #         st.caption("All non-routine maintenance")
    #     else:
    #         st.metric(label="📋 Maintenance Backlog", value="N/A")
    # VISUALIZATIONS SECTION
    st.markdown("---")
    st.header("Corrosion Analysis")

    # Row 3: Main Visualizations
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.subheader("Risk Distribution")
        if 'Risk_Level' in filtered_data.columns and not filtered_data.empty:
            risk_counts = filtered_data['Risk_Level'].value_counts()
            fig1 = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Pipeline Risk Levels Distribution",
                color=risk_counts.index,
                color_discrete_map={'Critical': '#FF4B4B', 'High': '#FFA500', 'Medium': '#FFD700', 'Low': '#90EE90'}
            )
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("No risk level data available")

    with viz_col2:
        st.subheader("Corrosion Trends")
        if 'time_years' in filtered_data.columns and 'corrosion_rate_mm_per_year' in filtered_data.columns and not filtered_data.empty:
            # Create age groups for trend analysis
            filtered_data_copy = filtered_data.copy()
            filtered_data_copy['age_group'] = pd.cut(filtered_data_copy['time_years'],
                                                     bins=[0, 5, 10, 15, 20, 30, 50],
                                                     labels=['0-5y', '5-10y', '10-15y', '15-20y', '20-30y', '30+y'])

            age_trends = filtered_data_copy.groupby('age_group')['corrosion_rate_mm_per_year'].mean().reset_index()

            fig2 = px.line(
                age_trends,
                x='age_group',
                y='corrosion_rate_mm_per_year',
                title="Corrosion Rate by Pipeline Age",
                markers=True
            )
            fig2.update_layout(
                xaxis_title="Pipeline Age Group",
                yaxis_title="Corrosion Rate (mm/year)"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No age or corrosion data available")

    # Row 4: Additional Visualizations
    viz_col3, viz_col4 = st.columns(2)

    with viz_col3:
        st.subheader("Material Performance")
        if 'material' in filtered_data.columns and 'corrosion_rate_mm_per_year' in filtered_data.columns and not filtered_data.empty:
            material_stats = filtered_data.groupby('material').agg({
                'corrosion_rate_mm_per_year': 'mean',
                'pipe_size_mm': 'count'
            }).reset_index()

            fig3 = px.bar(
                material_stats,
                x='material',
                y='corrosion_rate_mm_per_year',
                title="Average Corrosion Rate by Material",
                color='corrosion_rate_mm_per_year',
                color_continuous_scale='RdYlGn_r'  # Red for high corrosion, Green for low
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No material or corrosion data available")

    with viz_col4:
        st.subheader("Pressure vs Corrosion")
        if 'max_pressure_psi' in filtered_data.columns and 'corrosion_rate_mm_per_year' in filtered_data.columns and not filtered_data.empty:
            fig4 = px.scatter(
                filtered_data,
                x='max_pressure_psi',
                y='corrosion_rate_mm_per_year',
                color='Risk_Level' if 'Risk_Level' in filtered_data.columns else None,
                title="Operating Pressure vs Corrosion Rate",
                trendline="lowess"
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No pressure or corrosion data available")

    # DATA TABLE SECTION
    st.markdown("---")
    st.header("Pipeline Details")

    if not filtered_data.empty:
        # Select relevant columns to display
        display_columns = [
            'pipe_size_mm', 'material', 'time_years', 'max_pressure_psi',
            'corrosion_rate_mm_per_year', 'thickness_loss_mm', 'Risk_Level',
            'Maintenance_Recommendation', 'Prediction_Confidence'
        ]

        available_columns = [col for col in display_columns if col in filtered_data.columns]

        if available_columns:
            st.dataframe(
                filtered_data[available_columns],
                use_container_width=True,
                height=400
            )

            # Download button
            csv = filtered_data[available_columns].to_csv(index=False)
            st.download_button(
                label="Download Filtered Data",
                data=csv,
                file_name="filtered_pipeline_data.csv",
                mime="text/csv"
            )
        else:
            st.info("No display columns available in the filtered data")
    else:
        st.info("No data available with current filters")

# PAGE 2: CORROSION SIMULATOR
elif page == "Corrosion Simulator":
    import streamlit as st
    import pandas as pd
    import joblib
    import numpy as np
    from preprocessing import prepare_features, validate_input_ranges

    # --- Page Configuration ---
    st.set_page_config(
        page_title="Pipeline Corrosion Risk Dashboard",
        page_icon="🔧",
        layout="wide"
    )


    # --- Load the trained model ---
    @st.cache_resource
    def load_model():
        try:
            model = joblib.load("corrosion_voting_model.pkl")
            st.success("✅ Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None


    def get_confidence_analysis(confidence_score, user_input):
        """
        Analyze confidence score and provide insights
        """
        analysis = {
            'level': '',
            'color': '',
            'message': '',
            'suggestions': []
        }

        # Confidence thresholds
        if confidence_score >= 0.90:
            analysis['level'] = 'Very High'
            analysis['color'] = 'green'
            analysis['message'] = 'Model is highly confident in this prediction'
        elif confidence_score >= 0.75:
            analysis['level'] = 'High'
            analysis['color'] = 'blue'
            analysis['message'] = 'Good confidence level'
        elif confidence_score >= 0.60:
            analysis['level'] = 'Medium'
            analysis['color'] = 'orange'
            analysis['message'] = 'Moderate confidence - consider verifying inputs'
        else:
            analysis['level'] = 'Low'
            analysis['color'] = 'red'
            analysis['message'] = 'Low confidence - prediction may be unreliable'

        return analysis


    def get_actionable_insights(predicted_label, confidence_score, probabilities, user_input):
        """
        Generate actionable insights for Maintenance and Asset Integrity teams
        """
        insights = {
            'maintenance': [],
            'asset_integrity': [],
            'urgency': '',
            'timeline': ''
        }

        # Define risk levels and corresponding actions
        if predicted_label == "Normal":
            insights['urgency'] = 'Low'
            insights['timeline'] = 'Next scheduled maintenance'
            insights['maintenance'] = [
                "Continue routine inspection schedule",
                "Monitor corrosion rates during next planned outage",
                "Verify protective coatings are intact",
                "Document current condition for baseline reference"
            ]
            insights['asset_integrity'] = [
                "Include in annual integrity assessment",
                "Update risk matrix with current condition",
                "Review corrosion monitoring program effectiveness",
                "Plan for next comprehensive inspection cycle"
            ]

        elif predicted_label == "Moderate":
            insights['urgency'] = 'Medium'
            insights['timeline'] = 'Within 3-6 months'
            insights['maintenance'] = [
                "Schedule enhanced NDT inspection (UT thickness testing)",
                "Increase monitoring frequency to quarterly",
                "Prepare corrosion mitigation materials",
                "Review operating parameters for stress reduction opportunities"
            ]
            insights['asset_integrity'] = [
                "Update risk-based inspection plan",
                "Assess remaining life and fitness-for-service",
                "Evaluate need for corrosion inhibitors",
                "Consider inline inspection tool deployment"
            ]

        else:  # Critical
            insights['urgency'] = 'High'
            insights['timeline'] = 'Immediate action required'
            insights['maintenance'] = [
                "🚨 IMMEDIATE: Schedule shutdown for detailed inspection",
                "Implement temporary repairs if immediate shutdown not possible",
                "Increase monitoring to weekly or continuous",
                "Prepare emergency repair kit and team"
            ]
            insights['asset_integrity'] = [
                "🚨 Conduct immediate fitness-for-service assessment",
                "Evaluate replacement options and lead times",
                "Review design parameters for potential upgrades",
                "Update emergency response plans for this asset"
            ]

        # Add confidence-based recommendations
        if confidence_score < 0.7:
            insights['maintenance'].insert(0,
                                           "⚠️ Verify prediction with additional field inspection due to lower confidence")
            insights['asset_integrity'].insert(0, "⚠️ Consider additional assessment methods to confirm prediction")

        # Add material-specific recommendations
        material = user_input['material'].iloc[0].lower()
        if 'carbon' in material:
            insights['maintenance'].append("Focus on cathodic protection system verification")
            insights['asset_integrity'].append("Review corrosion allowance utilization")
        elif 'stainless' in material:
            insights['maintenance'].append("Check for chloride stress corrosion cracking")
            insights['asset_integrity'].append("Verify material certification and grade")

        # Add pressure-specific recommendations for high pressure systems
        if user_input['max_pressure_psi'].iloc[0] > 300:
            insights['maintenance'].append("Verify pressure relief systems are operational")
            insights['asset_integrity'].append("Review pressure cycle counting and fatigue analysis")

        return insights


    # --- Main Application ---
    def main():
        st.title("🏭 Pipeline Corrosion Risk Simulation Dashboard")
        st.markdown("""
        This tool predicts corrosion risk levels using our trained ensemble model.  
        """)

        # Load model
        model = load_model()
        if model is None:
            st.stop()

        # --- Input Section ---
        st.markdown("## 🧪 Input Parameters")
        st.caption("Adjust parameters below. Typical ranges are shown for reference.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📏 Physical Properties")
            pipe_size = st.slider("Pipe Size (mm)", 50, 500, 200, 10,
                                  help="Typical range: 100-400 mm")
            thickness = st.slider("Thickness (mm)", 3, 30, 8, 1,
                                  help="Typical range: 6-20 mm")
            material = st.selectbox("Material",
                                    ["Carbon Steel", "Fiberglass", "HDPE", "PVC", "Stainless Steel"],
                                    help="Material corrosion resistance")

        with col2:
            st.markdown("### ⚙️ Operational Conditions")
            max_pressure = st.slider("Max Pressure (psi)", 50, 1000, 200, 50,
                                     help="Typical range: 100-500 psi")
            temperature = st.slider("Temperature (°C)", 0, 150, 60, 5,
                                    help="Typical range: 20-100°C")
            time_years = st.slider("Service Time (Years)", 0, 50, 10, 1,
                                   help="Typical range: 0-30 years")

        with col3:
            st.markdown("### 📊 Corrosion Metrics")
            corrosion_impact = st.slider("Corrosion Impact (%)", 0, 50, 10, 1,
                                         help="Typical range: 0-25%")
            loss_percent = st.slider("Material Loss (%)", 0, 30, 5, 1,
                                     help="Typical range: 0-15%")

        # Use pipe_size for both original and transformed feature
        pipe_size_tf = pipe_size

        # --- Create input dataframe ---
        user_input = pd.DataFrame([{
            'pipe_size_mm': pipe_size,
            'thickness_mm': thickness,
            'material': material,
            'max_pressure_psi': max_pressure,
            'temperature_c': temperature,
            'corrosion_impact_percent': corrosion_impact,
            'material_loss_percent': loss_percent,
            'time_years': time_years,
            # REMOVE: 'Stress_Index': stress_index,  # This is now calculated!
            'pipe_size_mm_tf': pipe_size_tf
        }])

        # --- Prediction Section ---
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")

        if st.button("🎯 Run Prediction", type="primary"):
            with st.spinner("Processing prediction..."):
                try:
                    # Run validation internally (but don't display)
                    warnings = validate_input_ranges(user_input)

                    # Feature preparation
                    prepared_input = prepare_features(user_input.copy(), model=model)

                    # Make prediction
                    prediction = model.predict(prepared_input)[0]
                    probabilities = model.predict_proba(prepared_input)[0]
                    confidence_score = probabilities.max()

                    # Map prediction to label
                    condition_labels = {0: "Normal", 1: "Moderate", 2: "Critical"}
                    risk_colors = {
                        "Normal": "#4CAF50",
                        "Moderate": "#FFC107",
                        "Critical": "#F44336"
                    }

                    predicted_label = condition_labels.get(prediction, "Unknown")
                    color = risk_colors.get(predicted_label, "#666666")

                    # Confidence analysis
                    analysis = get_confidence_analysis(confidence_score, user_input)

                    # Get actionable insights
                    insights = get_actionable_insights(predicted_label, confidence_score, probabilities, user_input)

                    # --- COMPACT PREDICTION RESULT (Very compact) ---
                    st.markdown("### Assessment Result")

                    # Main prediction card - very compact
                    st.markdown(f"""
                    <div style="padding:20px; border-left: 5px solid {color}; background-color: #f8f9fa; border-radius: 8px; margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h2 style="margin:0; color:{color}; font-size:1.8em;">{predicted_label}</h2>
                                <p style="margin:5px 0; color:#666; font-size:1em;">
                                    Risk Level • {analysis['level']} reliability
                                </p>
                            </div>
                            <div style="text-align: center; background-color: white; padding: 8px 15px; border-radius: 6px; border: 1px solid #ddd;">
                                <p style="margin:0; font-size:0.8em; color:#666; font-weight: bold;">CONFIDENCE</p>
                                <p style="margin:0; font-size:1.2em; color:{color}; font-weight: bold;">{confidence_score:.1%}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # --- ENHANCED ACTIONABLE INSIGHTS SECTION ---
                    st.markdown("---")
                    st.markdown("## 🎯 Actionable Insights")

                    # Get actionable insights
                    insights = get_actionable_insights(predicted_label, confidence_score, probabilities, user_input)

                    # Professional metrics header
                    st.markdown("### 📈 Risk Assessment Overview")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;">
                            <div style="font-size: 0.9em; opacity: 0.9;">Risk Level</div>
                            <div style="font-size: 1.8em; font-weight: bold; margin: 10px 0;">{predicted_label}</div>
                            <div style="font-size: 0.8em; opacity: 0.9;">{insights['urgency']} urgency</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        # Define icon and label based on urgency level
                        if insights['urgency'] == 'Low':
                            timeline_icon = "🟢"
                            timeline_label = "Low Urgency"
                        elif insights['urgency'] == 'Medium':
                            timeline_icon = "🟡"
                            timeline_label = "Medium Urgency"
                        else:
                            timeline_icon = "🔴"
                            timeline_label = "High Urgency"

                        # Display styled box with both icon and label
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                                    padding: 20px; border-radius: 10px; color: white;">
                            <div style="font-size: 0.9em; opacity: 0.9;">Timeline</div>
                            <div style="display: flex; align-items: center; gap: 10px; margin: 10px 0;">
                                <span style="font-size: 1.8em;">{timeline_icon}</span>
                                <span style="font-size: 1.1em; font-weight: bold;">{timeline_label}</span>
                            </div>
                            <div style="font-size: 0.8em; opacity: 0.9;">{insights['timeline']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        confidence_color = "#00C851" if confidence_score > 0.8 else "#ffbb33" if confidence_score > 0.6 else "#ff4444"
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 10px; color: white;">
                            <div style="font-size: 0.9em; opacity: 0.9;">Confidence Score</div>
                            <div style="font-size: 1.8em; font-weight: bold; margin: 10px 0;">{confidence_score:.1%}</div>
                            <div style="font-size: 0.8em; opacity: 0.9;">{analysis['level']} reliability</div>
                        </div>
                        """, unsafe_allow_html=True)


                    # Horizontal Tab Selection
                    st.markdown("### User Group")
                    #tab1, tab2, tab3 = st.tabs(["🔧 Maintenance Team", "🏢 Asset Integrity Team", "👥 Both Teams"])
                    tab1, tab2 = st.tabs(["🔧 Maintenance Team", "🏢 Asset Integrity Team"])

                    with tab1:
                        st.markdown("### 🛠️ Maintenance Operations")

                        col_actions, col_metrics = st.columns([2, 1])

                        with col_actions:
                            st.markdown("**Key Responsibilities:**")
                            for i, action in enumerate(insights['maintenance'], 1):
                                st.markdown(f"• {action}")

                        with col_metrics:
                            st.markdown("**Performance Metrics**")

                            # Maintenance metrics in cards
                            maintenance_metrics = [
                                {"label": "Inspection Frequency",
                                 "value": "Annual" if predicted_label == "Normal" else "Quarterly" if predicted_label == "Moderate" else "Monthly",
                                 "icon": "📊"},
                                {"label": "Work Priority",
                                 "value": "Routine" if predicted_label == "Normal" else "High" if predicted_label == "Moderate" else "Emergency",
                                 "icon": "🎯"},
                                {"label": "Next NDT Due",
                                 "value": "Next outage" if predicted_label == "Normal" else "3 months" if predicted_label == "Moderate" else "Immediately",
                                 "icon": "⏰"}
                            ]

                            for metric in maintenance_metrics:
                                st.markdown(f"""
                                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; margin: 10px 0;">
                                    <div style="font-size: 0.8em; color: #6c757d;">{metric['icon']} {metric['label']}</div>
                                    <div style="font-size: 1.2em; font-weight: bold; color: #495057;">{metric['value']}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    with tab2:
                        st.markdown("### 💼 Asset Integrity Management")

                        col_strategy, col_impact = st.columns([2, 1])

                        with col_strategy:
                            st.markdown("**Strategic Responsibilities:**")
                            for i, action in enumerate(insights['asset_integrity'], 1):
                                st.markdown(f"• {action}")

                        with col_impact:
                            st.markdown("**Business Impact**")

                            # Integrity metrics in cards
                            integrity_metrics = [
                                {"label": "Risk Ranking",
                                 "value": "Low" if predicted_label == "Normal" else "Medium" if predicted_label == "Moderate" else "High",
                                 "icon": "📈"},
                                {"label": "Budget Impact",
                                 "value": "Low" if predicted_label == "Normal" else "Medium" if predicted_label == "Moderate" else "High",
                                 "icon": "💰"},
                                {"label": "Compliance",
                                 "value": "Compliant" if predicted_label == "Normal" else "Monitor" if predicted_label == "Moderate" else "Review",
                                 "icon": "⚖️"}
                            ]

                            for metric in integrity_metrics:
                                value_color = "#28a745" if metric['value'] == 'Low' else "#ffc107" if metric[
                                                                                                          'value'] == 'Medium' else "#dc3545"
                                st.markdown(f"""
                                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid {value_color}; margin: 10px 0;">
                                    <div style="font-size: 0.8em; color: #6c757d;">{metric['icon']} {metric['label']}</div>
                                    <div style="font-size: 1.2em; font-weight: bold; color: {value_color};">{metric['value']}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    # with tab3:
                    #     st.markdown("### 👥 Combined Team View")
                    #
                    #     # Quick stats
                    #     col_stat1, col_stat2, col_stat3 = st.columns(3)
                    #     with col_stat1:
                    #         st.markdown(f"""
                    #         <div style="text-align: center; padding: 15px; background: #e8f5e8; border-radius: 10px;">
                    #             <div style="font-size: 0.9em; color: #2e7d32;">Maintenance Tasks</div>
                    #             <div style="font-size: 1.5em; font-weight: bold; color: #2e7d32;">{len(insights['maintenance'])}</div>
                    #         </div>
                    #         """, unsafe_allow_html=True)
                    #
                    #     with col_stat2:
                    #         st.markdown(f"""
                    #         <div style="text-align: center; padding: 15px; background: #e3f2fd; border-radius: 10px;">
                    #             <div style="font-size: 0.9em; color: #1565c0;">Integrity Tasks</div>
                    #             <div style="font-size: 1.5em; font-weight: bold; color: #1565c0;">{len(insights['asset_integrity'])}</div>
                    #         </div>
                    #         """, unsafe_allow_html=True)
                    #
                    #     with col_stat3:
                    #         total_tasks = len(insights['maintenance']) + len(insights['asset_integrity'])
                    #         st.markdown(f"""
                    #         <div style="text-align: center; padding: 15px; background: #fff3e0; border-radius: 10px;">
                    #             <div style="font-size: 0.9em; color: #ef6c00;">Total Actions</div>
                    #             <div style="font-size: 1.5em; font-weight: bold; color: #ef6c00;">{total_tasks}</div>
                    #         </div>
                    #         """, unsafe_allow_html=True)

                        col_team1, col_team2 = st.columns(2)

                        with col_team1:
                            st.markdown("#### 🔧 Maintenance Priorities")
                            for i, action in enumerate(insights['maintenance'][:4], 1):
                                st.markdown(f"**{i}.** {action}")

                        with col_team2:
                            st.markdown("#### 🏢 Integrity Priorities")
                            for i, action in enumerate(insights['asset_integrity'][:4], 1):
                                st.markdown(f"**{i}.** {action}")

                    # Final recommendation
                    st.markdown("---")
                    st.markdown("### 🚀 Recommended Action Plan")

                    action_plans = {
                        'Normal': {
                            'color': '#28a745',
                            'icon': '✅',
                            'message': 'Continue with routine maintenance schedule and baseline documentation.'
                        },
                        'Moderate': {
                            'color': '#ffc107',
                            'icon': '⚠️',
                            'message': 'Schedule enhanced inspections and review operating parameters within 2 weeks.'
                        },
                        'Critical': {
                            'color': '#dc3545',
                            'icon': '🚨',
                            'message': 'Initiate immediate assessment and prepare emergency response procedures.'
                        }
                    }

                    plan = action_plans[predicted_label]
                    st.markdown(f"""
                    <div style="background: {plan['color']}; padding: 20px; border-radius: 10px; color: white;">
                        <div style="font-size: 1.2em; font-weight: bold;">
                            {plan['icon']} {predicted_label} Risk Protocol
                        </div>
                        <div style="margin-top: 10px; font-size: 1em;">
                            {plan['message']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    # Probability breakdown (always at bottom)
                    st.markdown("---")
                    st.markdown("#### 📊 Probability Distribution")
                    cols = st.columns(3)
                    for idx, (condition, prob) in enumerate(zip(['Normal', 'Moderate', 'Critical'], probabilities)):
                        with cols[idx]:
                            prob_color = "#4CAF50" if condition == "Normal" else "#FFC107" if condition == "Moderate" else "#F44336"
                            st.markdown(f"""
                                <div style="border:2px solid {prob_color};padding:10px;border-radius:10px;text-align:center;">
                                    <h4 style="margin:0;color:{prob_color};">{condition}</h4>
                                    <h3 style="margin:5px 0;color:{prob_color};">{prob:.1%}</h3>
                                </div>
                                """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")

        else:
            # Show initial state
            st.info("👆 Click **Run Prediction** to analyze your pipeline configuration")

            # Show typical values for reference
            with st.expander("📋 Typical Parameter Ranges (for reference)"):
                st.markdown("""
                    | Parameter | Typical Range | Notes |
                    |-----------|---------------|-------|
                    | Pipe Size | 100-400 mm | Larger pipes distribute stress differently |
                    | Thickness | 6-20 mm | Thinner walls are more vulnerable |
                    | Pressure | 100-500 psi | Higher pressure increases stress |
                    | Temperature | 20-100°C | Higher temps accelerate corrosion |
                    | Corrosion Impact | 0-25% | Values >25% are extreme cases |
                    | Material Loss | 0-15% | Loss >15% indicates severe damage |
                    | Service Time | 0-30 years | Older pipes have more accumulated damage |
                    """)

            # --- Footer ---
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; color: gray;'>
                <p>Built with Streamlit | Model: Voting Ensemble (XGBoost + LightGBM + Random Forest)</p>
            </div>
            """, unsafe_allow_html=True)


    if __name__ == "__main__":
        main()
# PAGE 3: ML MODEL PERFORMANCE
elif page == "ML Model Performance":
    import streamlit as st
    import pandas as pd
    import plotly.express as px

    # --- PAGE TITLE ---
    st.title("📈 Model Performance Comparison Dashboard")
    st.markdown("""
    Explore and compare different **machine learning models** used in corrosion risk forecasting.  
    Use the filters below to select which models and performance metrics you'd like to analyze.
    """)

    # --- MODEL PERFORMANCE DATA ---
    data = {
        "Model": [
            "Basic XGBoost",
            "Tuned XGBoost",
            "Voting (XGB + RF + LR)",
            "Voting (XGB + LGBM + RF)",
            "Voting (XGB + RF + SVM)",
            "Stacking (XGB + RF + SVM)",
            "Stacking (XGB + LGBM + RF)"
        ],
        "Accuracy": [0.356, 0.847, 0.847, 0.852, 0.839, 0.844, 0.847],
        "Precision": [0.275, 0.848, 0.847, 0.852, 0.839, 0.844, 0.847],
        "Recall": [0.286, 0.847, 0.847, 0.852, 0.839, 0.844, 0.847],
        "F1-Score": [0.278, 0.847, 0.847, 0.852, 0.839, 0.843, 0.847],
        "Log Loss": [1.675, 0.384, 0.611, 0.408, 0.615, 0.373, 0.377]
    }

    df = pd.DataFrame(data)

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🔍 Filters")
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        options=df["Model"].unique(),
        default=df["Model"].unique()
    )

    selected_metrics = st.sidebar.multiselect(
        "Select Metrics to Display",
        options=["Accuracy", "Precision", "Recall", "F1-Score", "Log Loss"],
        default=["Accuracy", "F1-Score", "Log Loss"]
    )

    # --- FILTER DATA BASED ON USER SELECTION ---
    filtered_df = df[df["Model"].isin(selected_models)]

    # --- SUMMARY TABLE ---
    st.markdown("### 🧾 Model Performance Summary")
    st.dataframe(filtered_df.style.highlight_max(axis=0, color='lightgreen'))

    # --- MELT DATA FOR VISUALIZATION ---
    melted = filtered_df.melt(id_vars="Model", value_vars=selected_metrics,
                              var_name="Metric", value_name="Score")

    # --- BAR CHART ---
    st.markdown("### 📊 Visual Comparison")
    fig = px.bar(
        melted,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text_auto=".3f",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        title="Model Performance Comparison (Including Log Loss)",
        xaxis_tickangle=15,
        legend_title_text='Metrics'
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- LINE CHART (optional if multiple metrics selected) ---
    if len(selected_metrics) > 2:
        st.markdown("### 📈 Metric Trend Line")
        fig2 = px.line(
            melted,
            x="Model",
            y="Score",
            color="Metric",
            markers=True,
            title="Performance Trend Across Models",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig2.update_layout(xaxis_tickangle=15)
        st.plotly_chart(fig2, use_container_width=True)

    # --- BEST MODEL SECTION ---
    st.markdown("---")
    best_model = df.loc[df['Accuracy'].idxmax()]
    st.markdown(f"""
    ### 🏆 Best Performing Model: **{best_model['Model']}**
    - **Accuracy:** {best_model['Accuracy']:.3f}  
    - **Precision:** {best_model['Precision']:.3f}  
    - **Recall:** {best_model['Recall']:.3f}  
    - **F1-Score:** {best_model['F1-Score']:.3f}  
    - **Log Loss:** {best_model['Log Loss']:.3f}

    💡 *This model offers the most balanced trade-off between accuracy, precision, recall, and log loss,  
    making it ideal for deployment in the corrosion risk forecasting dashboard.*
    """)

    # --- FOOTNOTE ---
    st.caption("""
    Use the filters on the left to explore how different models perform across various metrics.  
    Lower **Log Loss** indicates better probability calibration and fewer prediction errors.
    """)

elif page == "Bulk Pipeline Analysis":
    import streamlit as st
    import pandas as pd
    import joblib
    import numpy as np
    from preprocessing import prepare_features
    import io
    import base64
    from datetime import datetime

    # --- Page Configuration ---
    st.set_page_config(
        page_title="Bulk Pipeline Analysis",
        page_icon="📊",
        layout="wide"
    )


    # --- Load the trained model ---
    @st.cache_resource
    def load_model():
        try:
            model = joblib.load("corrosion_voting_model.pkl")
            return model
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None


    def validate_uploaded_data(df):
        """Validate the uploaded CSV has required columns (case insensitive)"""
        required_columns = [
            'pipe_size_mm', 'thickness_mm', 'material', 'max_pressure_psi',
            'temperature_c', 'corrosion_impact_percent', 'material_loss_percent', 'time_years'
        ]

        # Convert both required columns and actual columns to lowercase for comparison
        df_columns_lower = [col.lower().strip() for col in df.columns]
        required_columns_lower = [col.lower() for col in required_columns]

        missing_columns = []

        for i, req_col in enumerate(required_columns_lower):
            if req_col not in df_columns_lower:
                missing_columns.append(required_columns[i])  # Append original case for display

        return missing_columns


    def process_batch_prediction(df, model):
        """Process batch prediction on uploaded data (case insensitive)"""
        try:
            # Create a mapping from lowercase to original column names
            column_mapping = {col.lower().strip(): col for col in df.columns}

            # Standardize column names to lowercase for processing
            df_standardized = df.rename(columns=lambda x: x.lower().strip())

            # Prepare features
            prepared_data = prepare_features(df_standardized.copy(), model=model)

            # Make predictions
            predictions = model.predict(prepared_data)
            probabilities = model.predict_proba(prepared_data)

            # Add results to dataframe (using original column names)
            df_result = df.copy()  # Keep original column names
            df_result['Predicted_Condition'] = predictions
            df_result['Predicted_Condition_Label'] = ['Normal' if x == 0 else 'Moderate' if x == 1 else 'Critical' for x
                                                      in predictions]
            df_result['Probability_Normal'] = probabilities[:, 0]
            df_result['Probability_Moderate'] = probabilities[:, 1]
            df_result['Probability_Critical'] = probabilities[:, 2]
            df_result['Prediction_Confidence'] = probabilities.max(axis=1)
            df_result['Risk_Level'] = df_result['Predicted_Condition_Label'].map({
                'Normal': 'Low Risk',
                'Moderate': 'Medium Risk',
                'Critical': 'High Risk'
            })

            return df, None

        except Exception as e:
            return None, str(e)


    def get_download_link(df, filename="pipeline_predictions.csv"):
        """Generate a download link for the processed dataframe"""
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Predictions CSV</a>'
        return href


    def main():
        st.title("📊 Bulk Pipeline Analysis")
        st.markdown("""
        Upload a CSV file containing multiple pipeline records to get comprehensive risk assessments 
        for your entire pipeline fleet. Analyze trends, identify high-risk assets, and export results for further analysis.
        """)

        # Load model
        model = load_model()
        if model is None:
            st.stop()

        # File upload section
        st.markdown("---")
        st.markdown("## 📁 Upload Pipeline Data")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file with pipeline data. Required columns: pipe_size_mm, thickness_mm, material, max_pressure_psi, temperature_c, corrosion_impact_percent, material_loss_percent, time_years"
            )

        with col2:
            st.markdown("""
            ### 📋 Required Columns
            - `pipe_size_mm`
            - `thickness_mm` 
            - `material`
            - `max_pressure_psi`
            - `temperature_c`
            - `corrosion_impact_percent`
            - `material_loss_percent`
            - `time_years`
            """)

        # Sample data download
        with st.expander("📥 Download Sample Template"):
            sample_data = pd.DataFrame({
                'pipe_size_mm': [200, 150, 300, 250, 180],
                'thickness_mm': [8, 10, 12, 9, 7],
                'material': ['carbon steel', 'stainless steel', 'fiberglass', 'carbon steel', 'pvc'],
                'max_pressure_psi': [200, 150, 180, 220, 120],
                'temperature_c': [60, 45, 75, 55, 40],
                'corrosion_impact_percent': [5, 8, 3, 12, 2],
                'material_loss_percent': [2, 4, 1, 6, 1],
                'time_years': [10, 5, 8, 15, 3]
            })
            st.dataframe(sample_data)
            st.markdown(get_download_link(sample_data, "pipeline_data_template.csv"), unsafe_allow_html=True)

        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)

                st.markdown("---")
                st.markdown("## 🔍 Data Preview")

                # Show basic info
                col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                with col_info1:
                    st.metric("Total Pipelines", len(df))
                with col_info2:
                    st.metric("Columns", len(df.columns))
                with col_info3:
                    st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
                with col_info4:
                    material_types = df['material'].nunique() if 'material' in df.columns else 0
                    st.metric("Material Types", material_types)

                # Show data preview
                st.dataframe(df.head(10), use_container_width=True)

                # Validate data
                st.markdown("---")
                st.markdown("## ✅ Data Validation")

                missing_columns = validate_uploaded_data(df)

                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                    st.info(
                        "Please ensure your CSV contains all required columns. Download the sample template above for reference.")
                else:
                    st.success("✅ All required columns are present!")

                    # Process predictions
                    if st.button("🚀 Analyze Pipeline Fleet", type="primary", use_container_width=True):
                        with st.spinner("Processing bulk analysis..."):
                            result_df, error = process_batch_prediction(df, model)

                            if error:
                                st.error(f"❌ Analysis failed: {error}")
                            else:
                                st.success(f"✅ Successfully analyzed {len(result_df)} pipelines!")

                                # Results overview
                                st.markdown("---")
                                st.markdown("## 📊 Fleet Analysis Summary")

                                # Summary statistics
                                col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)

                                with col_sum1:
                                    normal_count = len(result_df[result_df['Predicted_Condition_Label'] == 'Normal'])
                                    normal_percent = (normal_count / len(result_df)) * 100
                                    st.metric("Low Risk Pipelines", f"{normal_count}",
                                              f"{normal_percent:.1f}%")

                                with col_sum2:
                                    moderate_count = len(
                                        result_df[result_df['Predicted_Condition_Label'] == 'Moderate'])
                                    moderate_percent = (moderate_count / len(result_df)) * 100
                                    st.metric("Medium Risk Pipelines", f"{moderate_count}",
                                              f"{moderate_percent:.1f}%")

                                with col_sum3:
                                    critical_count = len(
                                        result_df[result_df['Predicted_Condition_Label'] == 'Critical'])
                                    critical_percent = (critical_count / len(result_df)) * 100
                                    st.metric("High Risk Pipelines", f"{critical_count}",
                                              f"{critical_percent:.1f}%")

                                with col_sum4:
                                    avg_confidence = result_df['Prediction_Confidence'].mean()
                                    st.metric("Average Confidence", f"{avg_confidence:.1%}")

                                # Risk distribution chart
                                st.markdown("### 📈 Risk Distribution")
                                risk_counts = result_df['Predicted_Condition_Label'].value_counts()

                                col_chart1, col_chart2, col_chart3 = st.columns(3)

                                with col_chart1:
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;">
                                        <div style="font-size: 2em; font-weight: bold;">{normal_count}</div>
                                        <div style="font-size: 1em;">Low Risk</div>
                                        <div style="font-size: 0.8em; opacity: 0.9;">{normal_percent:.1f}% of fleet</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                with col_chart2:
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #FFC107 0%, #ffb300 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;">
                                        <div style="font-size: 2em; font-weight: bold;">{moderate_count}</div>
                                        <div style="font-size: 1em;">Medium Risk</div>
                                        <div style="font-size: 0.8em; opacity: 0.9;">{moderate_percent:.1f}% of fleet</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                with col_chart3:
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #F44336 0%, #d32f2f 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;">
                                        <div style="font-size: 2em; font-weight: bold;">{critical_count}</div>
                                        <div style="font-size: 1em;">High Risk</div>
                                        <div style="font-size: 0.8em; opacity: 0.9;">{critical_percent:.1f}% of fleet</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                # Detailed results
                                st.markdown("---")
                                st.markdown("## 🔍 Detailed Analysis Results")

                                # Filters for the results
                                col_filter1, col_filter2, col_filter3 = st.columns(3)

                                with col_filter1:
                                    risk_filter = st.selectbox(
                                        "Filter by Risk Level",
                                        ["All", "Low Risk", "Medium Risk", "High Risk"]
                                    )

                                with col_filter2:
                                    confidence_threshold = st.slider(
                                        "Minimum Confidence",
                                        min_value=0.0,
                                        max_value=1.0,
                                        value=0.7,
                                        step=0.1
                                    )

                                with col_filter3:
                                    sort_by = st.selectbox(
                                        "Sort by",
                                        ["Risk Level", "Confidence", "Corrosion Impact"]
                                    )

                                # Apply filters
                                filtered_df = result_df.copy()
                                if risk_filter != "All":
                                    risk_map = {"Low Risk": "Normal", "Medium Risk": "Moderate",
                                                "High Risk": "Critical"}
                                    filtered_df = filtered_df[
                                        filtered_df['Predicted_Condition_Label'] == risk_map[risk_filter]]

                                filtered_df = filtered_df[filtered_df['Prediction_Confidence'] >= confidence_threshold]

                                if sort_by == "Risk Level":
                                    filtered_df = filtered_df.sort_values('Predicted_Condition', ascending=False)
                                elif sort_by == "Confidence":
                                    filtered_df = filtered_df.sort_values('Prediction_Confidence', ascending=False)
                                elif sort_by == "Corrosion Impact":
                                    filtered_df = filtered_df.sort_values('corrosion_impact_percent', ascending=False)

                                st.dataframe(filtered_df, use_container_width=True)

                                # Export section
                                st.markdown("---")
                                st.markdown("## 📤 Export Results")

                                col_export1, col_export2 = st.columns(2)

                                with col_export1:
                                    st.markdown("### Download Full Results")
                                    st.markdown(get_download_link(result_df,
                                                                  f"pipeline_fleet_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"),
                                                unsafe_allow_html=True)

                                with col_export2:
                                    st.markdown("### Download Filtered Results")
                                    st.markdown(get_download_link(filtered_df,
                                                                  f"filtered_pipeline_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"),
                                                unsafe_allow_html=True)

                                # High-risk alert
                                if critical_count > 0:
                                    st.markdown("---")
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #F44336 0%, #d32f2f 100%); padding: 20px; border-radius: 10px; color: white;">
                                        <h3 style="margin:0;">🚨 Priority Alert</h3>
                                        <p style="margin:10px 0 0 0;">
                                            {critical_count} pipeline(s) require immediate attention. 
                                            Review high-risk assets and initiate inspection protocols.
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

        else:
            # Show features when no file is uploaded
            st.markdown("---")
            st.markdown("## 🚀 Bulk Analysis Features")

            col_feat1, col_feat2, col_feat3 = st.columns(3)

            with col_feat1:
                st.markdown("""
                ### 📈 Fleet Overview
                - Risk distribution across all pipelines
                - Priority asset identification
                - Confidence level analysis
                """)

            with col_feat2:
                st.markdown("""
                ### 🔍 Advanced Filtering
                - Filter by risk level
                - Sort by confidence scores
                - Material-specific analysis
                """)

            with col_feat3:
                st.markdown("""
                ### 📤 Export Capabilities
                - Download full results
                - Export filtered datasets
                - Timestamped reports
                """)


    if __name__ == "__main__":
        main()
