import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from who_data import calculate_percentile, weight_for_age_boys, weight_for_age_girls, height_for_age_boys, height_for_age_girls

def init_session_state():
    if 'growth_measurements' not in st.session_state:
        st.session_state.growth_measurements = []

def render_growth_tracker(food_df):
    init_session_state()
    
    st.markdown("## 📈 Growth Tracker")
    st.markdown("##### Track your baby's growth against WHO standards")
    st.markdown("---")

    # Input Section
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        
        with c1:
            baby_name = st.text_input("Baby Name", "My Baby")
            sex = st.selectbox("Sex", ["Boy", "Girl"])
            dob = st.date_input("Date of Birth", datetime.date.today() - datetime.timedelta(days=180))
            
        with c2:
            meas_weight = st.number_input("Current Weight (kg)", min_value=1.0, max_value=25.0, value=7.5, step=0.1)
            meas_height = st.number_input("Current Height/Length (cm)", min_value=30.0, max_value=100.0, value=65.0, step=0.5)
            meas_date = st.date_input("Measurement Date", datetime.date.today())
            
        # Calculate age in months
        age_days = (meas_date - dob).days
        age_months = age_days / 30.44
        
        # Prevent negative age or age > 24 for WHO 24-month tables
        if age_months < 0:
            st.warning("Measurement date cannot be before Date of Birth.")
            age_months = 0
            
        if st.button("Add Measurement", type="primary", use_container_width=True):
            if age_months > 24:
                st.error("This tracker currently supports WHO standards up to 24 months of age.")
            else:
                # Calculate percentiles
                w_perc, w_z, w_status = calculate_percentile(meas_weight, age_months, sex, "weight")
                h_perc, h_z, h_status = calculate_percentile(meas_height, age_months, sex, "height")
                
                new_meas = {
                    "Date": meas_date.strftime("%Y-%m-%d"),
                    "Age (months)": round(age_months, 1),
                    "Weight (kg)": meas_weight,
                    "Height (cm)": meas_height,
                    "Weight %ile": round(w_perc, 1),
                    "Height %ile": round(h_perc, 1),
                    "Status": w_status
                }
                
                # Check for duplicates and update or append
                st.session_state.growth_measurements.append(new_meas)
                # Sort by age
                st.session_state.growth_measurements = sorted(st.session_state.growth_measurements, key=lambda x: x["Age (months)"])
                st.success("Measurement added successfully!")
                
        st.markdown('</div>', unsafe_allow_html=True)

    if not st.session_state.growth_measurements:
        st.info("Add a measurement above to see your baby's growth summary and chart.")
        return

    # Use the most recent measurement for the summary cards
    latest = st.session_state.growth_measurements[-1]
    
    # Color badge logic
    def get_badge_color(percentile):
        if percentile < 15:
            return "#E74C3C", "Below average"
        elif percentile > 85:
            return "#F0A500", "Above average"
        return "#06D6A0", "Normal range"

    w_color, w_text = get_badge_color(latest["Weight %ile"])
    h_color, h_text = get_badge_color(latest["Height %ile"])

    # Summary Cards
    st.markdown("<br>", unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    
    with sc1:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:15px 10px;">
          <div style="font-size:.75rem;color:#999;font-weight:700;text-transform:uppercase;">Weight %ile</div>
          <div style="font-size:2.2rem;font-weight:900;color:{w_color};line-height:1.2;">{latest["Weight %ile"]}</div>
          <div style="background:{w_color}18;border-radius:20px;padding:2px 8px;font-size:.75rem;color:{w_color};font-weight:700;margin-top:5px;display:inline-block;">{w_text}</div>
        </div>""", unsafe_allow_html=True)
        
    with sc2:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:15px 10px;">
          <div style="font-size:.75rem;color:#999;font-weight:700;text-transform:uppercase;">Height %ile</div>
          <div style="font-size:2.2rem;font-weight:900;color:{h_color};line-height:1.2;">{latest["Height %ile"]}</div>
          <div style="background:{h_color}18;border-radius:20px;padding:2px 8px;font-size:.75rem;color:{h_color};font-weight:700;margin-top:5px;display:inline-block;">{h_text}</div>
        </div>""", unsafe_allow_html=True)
        
    with sc3:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:15px 10px;">
          <div style="font-size:.75rem;color:#999;font-weight:700;text-transform:uppercase;">Age</div>
          <div style="font-size:2.2rem;font-weight:900;color:#3D3D3D;line-height:1.2;">{latest["Age (months)"]}</div>
          <div style="font-size:.8rem;color:#999;margin-top:5px;">Months</div>
        </div>""", unsafe_allow_html=True)
        
    with sc4:
        status_color = "#E74C3C" if "Underweight" in latest["Status"] else "#F0A500" if "Overweight" in latest["Status"] else "#06D6A0"
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:15px 10px;">
          <div style="font-size:.75rem;color:#999;font-weight:700;text-transform:uppercase;">Growth Status</div>
          <div style="font-size:1.5rem;font-weight:900;color:{status_color};line-height:1.2;margin-top:10px;">{latest["Status"]}</div>
        </div>""", unsafe_allow_html=True)

    # Plotly Growth Chart
    st.markdown("<br>**📊 Growth Chart**", unsafe_allow_html=True)
    metric_toggle = st.radio("Display Metric", ["Weight (kg)", "Height (cm)"], horizontal=True, label_visibility="collapsed")
    
    who_df = None
    if metric_toggle == "Weight (kg)":
        who_df = weight_for_age_boys if sex == "Boy" else weight_for_age_girls
        y_col = "Weight (kg)"
    else:
        who_df = height_for_age_boys if sex == "Boy" else height_for_age_girls
        y_col = "Height (cm)"
        
    fig = go.Figure()
    
    # WHO Bands
    fig.add_trace(go.Scatter(x=who_df['age_months'], y=who_df['P3'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=who_df['age_months'], y=who_df['P15'], fill='tonexty', mode='lines', line=dict(width=0), fillcolor='rgba(255, 229, 229, 0.5)', name='< P15 (Below average)'))
    
    fig.add_trace(go.Scatter(x=who_df['age_months'], y=who_df['P15'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=who_df['age_months'], y=who_df['P85'], fill='tonexty', mode='lines', line=dict(width=0), fillcolor='rgba(229, 255, 229, 0.5)', name='P15-P85 (Normal)'))
    
    fig.add_trace(go.Scatter(x=who_df['age_months'], y=who_df['P85'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=who_df['age_months'], y=who_df['P97'], fill='tonexty', mode='lines', line=dict(width=0), fillcolor='rgba(255, 245, 229, 0.5)', name='> P85 (Above average)'))
    
    # Median Line
    fig.add_trace(go.Scatter(x=who_df['age_months'], y=who_df['P50'], mode='lines', line=dict(color='rgba(0, 0, 0, 0.2)', width=1, dash='dash'), name='Median (P50)'))
    
    # Baby's Measurements
    meas_df = pd.DataFrame(st.session_state.growth_measurements)
    fig.add_trace(go.Scatter(
        x=meas_df['Age (months)'], 
        y=meas_df[y_col], 
        mode='lines+markers', 
        line=dict(color='#FF6B6B', width=3), 
        marker=dict(size=10, color='#FF6B6B', line=dict(width=2, color='white')),
        name=f"{baby_name}'s {metric_toggle.split(' ')[0]}"
    ))
    
    # Current Age Line
    fig.add_vline(x=latest['Age (months)'], line_width=1, line_dash="dash", line_color="#4ECDC4")

    fig.update_layout(
        xaxis_title="Age (months)",
        yaxis_title=metric_toggle,
        paper_bgcolor="#fff",
        plot_bgcolor="#fff",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    fig.update_xaxes(showgrid=False, zeroline=False, range=[0, min(24, max(12, latest['Age (months)'] + 3))])
    fig.update_yaxes(showgrid=False, zeroline=False)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # Nutritional Recommendations
    st.markdown("<br>**🥗 Nutrition Recommendations for Growth**", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    if latest["Weight %ile"] < 15:
        st.error("📉 **Your baby may need calorie-dense foods.** Based on the weight percentile, consider increasing caloric and protein intake.")
        # Filter top 5 foods by cal+protein where quality >= 1 (Good/Excellent)
        recs = food_df[food_df['quality'] >= 1].copy()
        recs['score'] = recs['calories_kcal'] + (recs['protein_g'] * 4) # approx protein cals
        recs = recs.sort_values('score', ascending=False).head(5)
        st.markdown("**Recommended high-energy foods:**")
        
    elif latest["Weight %ile"] > 85:
        st.warning("📈 **Focus on nutrient-dense, lower-calorie options.** Ensure baby is getting plenty of vitamins and fiber without excessive calories.")
        recs = food_df[food_df['quality'] >= 1].copy()
        recs['score'] = recs['fiber_g'] + (recs['vitA_g'] * 10) - (recs['calories_kcal'] / 100)
        recs = recs.sort_values('score', ascending=False).head(5)
        st.markdown("**Recommended nutrient-dense foods:**")
        
    else:
        st.success("✅ **Great growth! Maintain balanced nutrition.** Your baby is tracking well within normal parameters.")
        recs = food_df[food_df['quality'] >= 1].copy()
        recs = recs.sample(min(3, len(recs))) # random balanced recommendations
        st.markdown("**Recommended balanced foods:**")
    
    # Display recommended foods
    for _, row in recs.iterrows():
        food_name = row.get('food_name', f"Food #{row.name}")
        cals = row.get('calories_kcal', 0)
        prot = row.get('protein_g', 0)
        fiber = row.get('fiber_g', 0)
        
        st.markdown(f"""
        <div style="background:#FAFAFA;border-radius:8px;padding:10px 15px;margin-bottom:8px;border-left:3px solid #4ECDC4;">
            <div style="font-weight:700;">{food_name}</div>
            <div style="font-size:0.8rem;color:#666;">🔥 {cals:.1f} kcal &nbsp;|&nbsp; 💪 {prot:.1f}g protein &nbsp;|&nbsp; 🌾 {fiber:.1f}g fiber</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

    # History Table
    st.markdown("<br>**📋 Measurement History**", unsafe_allow_html=True)
    
    st.dataframe(meas_df, use_container_width=True, hide_index=True)
    
    csv = meas_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download History CSV",
        csv,
        "baby_growth_history.csv",
        "text/csv",
        use_container_width=True
    )
    
    st.markdown("""
    <div style="margin-top:20px;font-size:0.75rem;color:#aaa;text-align:center;">
        <b>Note:</b> Growth percentiles and z-scores are calculated based on the official <i>WHO Child Growth Standards (2006)</i> for infants 0-24 months. 
        Calculations use interpolated LMS standard deviation scores.
    </div>
    """, unsafe_allow_html=True)
