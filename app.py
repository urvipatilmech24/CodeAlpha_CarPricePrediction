import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="CarValuate | AI Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- DARK BLUE GLASSMORPHISM CSS -----------------
st.markdown("""
<style>
    /* Global Background: Deep Midnight Navy Gradient */
    .stApp {
        background: radial-gradient(circle at 15% 20%, #0d1b2a 0%, #0a1128 50%, #03071e 100%) !important;
        color: #e0e6ed !important;
        font-family: 'Segoe UI', Inter, Roboto, sans-serif;
    }

    /* Transparent Frosted Glass Cards */
    .glass-card {
        background: rgba(16, 32, 60, 0.45);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.25s ease, border-color 0.25s ease;
        margin-bottom: 20px;
    }
    
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.45);
        transform: translateY(-2px);
    }

    /* Metric Glass Badges */
    .metric-badge {
        background: rgba(22, 43, 77, 0.4);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #38bdf8;
    }

    /* Hero Prediction Glow Card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(30, 58, 138, 0.25) 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 40px -10px rgba(14, 165, 233, 0.3);
    }
    .prediction-title {
        color: #94a3b8;
        font-size: 0.95rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .prediction-amount {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }

    /* Sidebar Glass Styling */
    [data-testid="stSidebar"] {
        background: rgba(10, 20, 40, 0.65) !important;
        backdrop-filter: blur(18px);
        border-right: 1px solid rgba(56, 189, 248, 0.12);
    }

    /* Custom Input Element Backgrounds */
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] {
        background-color: rgba(13, 27, 50, 0.7) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }

    /* Glass Action Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
        color: #ffffff;
        font-weight: 600;
        font-size: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 0.65rem 1.8rem;
        width: 100%;
        box-shadow: 0 4px 20px rgba(2, 132, 199, 0.35);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%);
        box-shadow: 0 6px 25px rgba(14, 165, 233, 0.5);
        transform: translateY(-1px);
        color: #ffffff;
    }
    /* Make the top Streamlit header completely transparent */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Optional: Hide the Deploy button entirely to make the UI look like a real production site */
    .stDeployButton {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- MODEL PIPELINE (CACHED) -----------------
@st.cache_resource
def load_and_train():
    df = pd.read_csv('car data.csv')
    df['Car_Age'] = 2024 - df['Year']
    df_clean = df.drop(columns=['Car_Name', 'Year'])
    df_encoded = pd.get_dummies(df_clean, drop_first=True)
    
    X = df_encoded.drop(columns=['Selling_Price'])
    y = df_encoded['Selling_Price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, X.columns.tolist(), r2, mae, rmse, df

model, feature_names, r2, mae, rmse, raw_df = load_and_train()

# ----------------- HEADER & METRICS ROW -----------------
st.markdown("""
<div style="margin-bottom: 25px;">
    <h1 style="color: #f8fafc; font-weight: 800; margin-bottom: 4px; font-size: 2.4rem;">
        AI Car Valuation Engine
    </h1>
    <p style="color: #94a3b8; font-size: 1rem; margin-top: 0;">
        Interactive machine learning estimator trained on market resale records
    </p>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="metric-badge">
        <div class="metric-label">Algorithm</div>
        <div class="metric-value" style="font-size: 1.25rem;">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-badge">
        <div class="metric-label">R² Accuracy</div>
        <div class="metric-value">{r2 * 100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-badge">
        <div class="metric-label">Mean Absolute Error</div>
        <div class="metric-value">₹{mae:.2f} L</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-badge">
        <div class="metric-label">RMSE Loss</div>
        <div class="metric-value">₹{rmse:.2f} L</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ----------------- SIDEBAR: SPECIFICATION CONTROLS -----------------
with st.sidebar:
    st.markdown("""
    <h2 style="color: #38bdf8; font-size: 1.3rem; margin-bottom: 2px;">Parameters</h2>
    <p style="color: #64748b; font-size: 0.85rem; margin-bottom: 20px;">Configure vehicle parameters</p>
    """, unsafe_allow_html=True)
    
    present_price = st.number_input(
        "Showroom / Ex-Showroom Price (Lakhs ₹)",
        min_value=0.5, max_value=95.0, value=6.5, step=0.25
    )
    kms_driven = st.slider("Total Kilometers Run", 500, 250000, 32000, step=1000)
    purchase_year = st.slider("Registration Year", 2005, 2024, 2016)
    
    st.markdown("<hr style='border: 0.5px solid rgba(56, 189, 248, 0.1); margin: 15px 0;'>", unsafe_allow_html=True)
    
    fuel_type = st.radio("Fuel Type", ["Petrol", "Diesel", "CNG"], horizontal=True)
    seller_type = st.radio("Seller Channel", ["Dealer", "Individual"], horizontal=True)
    transmission = st.radio("Transmission Type", ["Manual", "Automatic"], horizontal=True)
    owners = st.selectbox("Previous Owners", [0, 1, 2, 3], index=0)

# Build feature vector
car_age = 2024 - purchase_year
input_dict = {
    'Present_Price': present_price,
    'Driven_kms': kms_driven,
    'Owner': owners,
    'Car_Age': car_age,
    'Fuel_Type_Diesel': 1 if fuel_type == "Diesel" else 0,
    'Fuel_Type_Petrol': 1 if fuel_type == "Petrol" else 0,
    'Selling_type_Individual': 1 if seller_type == "Individual" else 0,
    'Transmission_Manual': 1 if transmission == "Manual" else 0
}
input_df = pd.DataFrame([input_dict])[feature_names]

# ----------------- MAIN INTERACTIVE TABS -----------------
tab_predict, tab_analytics, tab_data = st.tabs(["🎯 Valuation Engine", "📊 Model Insights", "📑 Raw Records"])

with tab_predict:
    col_left, col_right = st.columns([1.1, 1.4], gap="medium")
    
    with col_left:
        with st.container(border=True):
            st.markdown("<h3 style='color: #f1f5f9; margin-top:0;'>Run Valuation</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color: #94a3b8; font-size: 0.9rem;'>Evaluates your configured inputs through the trained regression forest.</p>", unsafe_allow_html=True)
            
            # Check for button click before running prediction
            if st.button("Generate Price Estimate ⚡"):
                predicted_val = model.predict(input_df)[0]
                predicted_val = max(0.1, predicted_val)  # Floor price
                depreciation = max(0, 100 - (predicted_val / present_price * 100))
                
                st.markdown(f"""
                <div class="prediction-card" style="margin-top: 15px;">
                    <div class="prediction-title">Predicted Market Value</div>
                    <div class="prediction-amount">₹ {predicted_val:.2f} Lakhs</div>
                    <div style="color: #94a3b8; font-size: 0.85rem;">
                        Depreciation factor: ~{depreciation:.1f}% from original price
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Configure your vehicle parameters and click the button above to calculate the valuation.")
        
    with col_right:
        with st.container(border=True):
            st.markdown("<h3 style='color: #f1f5f9; margin-top:0;'>Vehicle Summary Card</h3>", unsafe_allow_html=True)
            
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(f"**Showroom Price:** ₹ {present_price:.2f} L")
                st.markdown(f"**Mileage / Driven:** {kms_driven:,} km")
                st.markdown(f"**Calculated Age:** {car_age} years")
            with sc2:
                st.markdown(f"**Fuel Type:** {fuel_type}")
                st.markdown(f"**Transmission:** {transmission}")
                st.markdown(f"**Seller Category:** {seller_type}")
                
            st.markdown("<hr style='border: 0.5px solid rgba(56, 189, 248, 0.1); margin: 15px 0;'>", unsafe_allow_html=True)
            st.caption("Tip: Showroom price and vehicle age have the largest regression weights on the predicted resale outcome.")

with tab_analytics:
    st.container(border=True)
    st.markdown("<h3 style='color: #f1f5f9; margin-top:0;'>Feature Importance & Impact Weight</h3>", unsafe_allow_html=True)
    
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_alpha(0.0)      # Transparent figure background
    ax.set_facecolor('none')      # Fully transparent axes background
    
    y_pos = np.arange(len(importances))
    bars = ax.barh(y_pos, importances.values, color='#0284c7', edgecolor='#38bdf8', height=0.55, alpha=0.85)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importances.index, color='#cbd5e1', fontsize=10)
    ax.tick_params(colors='#94a3b8')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#38bdf8')
        spine.set_alpha(0.2)
        
    ax.grid(axis='x', linestyle='--', alpha=0.15, color='#38bdf8')
    ax.set_xlabel('Relative Weight / Gini Gain', color='#94a3b8', fontsize=10)
    plt.tight_layout()
    
    st.pyplot(fig)

with tab_data:
    st.container(border=True)
    st.markdown("<h3 style='color: #f1f5f9; margin-top:0;'>Training Dataset Preview</h3>", unsafe_allow_html=True)
    st.dataframe(raw_df.head(20), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
