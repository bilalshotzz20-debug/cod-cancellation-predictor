import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# =====================================================
# 1. PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="NEXUS AI | Enterprise",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# 2. ULTRA PREMIUM CSS (2026 DESIGN SYSTEM)
# =====================================================
st.markdown("""
<style>
/* --- FONTS & VARIABLES --- */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg-color: #030305;
    --card-bg: rgba(20, 20, 25, 0.6);
    --card-border: rgba(255, 255, 255, 0.08);
    --primary: #6366F1;
    --secondary: #06B6D4;
    --text-main: #F8FAFC;
    --text-muted: #94A3B8;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* --- BACKGROUND WITH MESH GRADIENT --- */
.stApp {
    background-color: var(--bg-color);
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(6, 182, 212, 0.15) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(139, 92, 246, 0.1) 0px, transparent 50%);
    color: var(--text-main);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1400px; padding: 2rem 2rem; }

/* --- GLASSMORPHISM CARDS --- */
.glass-card {
    background: var(--card-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--card-border);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.5);
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.2);
    transform: translateY(-2px);
}

/* --- TYPOGRAPHY --- */
.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    background: linear-gradient(135deg, #FFF 0%, #A5B4FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-muted);
    margin-top: 10px;
    font-weight: 400;
    max-width: 600px;
}

.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* --- BUTTONS --- */
.stButton > button {
    background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
    color: white;
    font-weight: 600;
    font-size: 1rem;
    padding: 12px 24px;
    border-radius: 12px;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
    color: white;
}

/* --- INPUT FIELDS --- */
.stSelectbox div[data-baseweb="select"], .stNumberInput div[data-baseweb="input"] {
    background-color: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    color: white !important;
}

.stSelectbox div[data-baseweb="select"]:focus-within, .stNumberInput div[data-baseweb="input"]:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
}

/* --- METRICS --- */
div[data-testid="stMetric"] {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(10px);
}

div[data-testid="stMetricValue"] {
    font-size: 2.5rem;
    font-weight: 800;
    color: white;
    background: linear-gradient(135deg, #FFF 0%, #A5B4FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

div[data-testid="stMetricLabel"] {
    color: var(--text-muted);
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. TRAIN MODEL (Cached)
# =====================================================
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 5000
    cities = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Kahuta", "Faisalabad", "Multan", "Peshawar"]
    
    df = pd.DataFrame({
        "city": np.random.choice(cities, n),
        "order_value": np.random.randint(500, 15001, n),
        "customer_age_days": np.random.randint(1, 366, n),
        "num_items": np.random.randint(1, 6, n),
        "delivery_days": np.random.randint(1, 11, n)
    })

    # Realistic Business Logic for Labels
    logit = (-3.0 + 0.65 * (df["delivery_days"] - 3) - 0.003 * np.minimum(df["customer_age_days"], 180) + 0.000025 * (df["order_value"] - 2500))
    prob = 1 / (1 + np.exp(-logit))
    df["is_cancelled"] = np.random.binomial(1, prob)

    encoded = pd.get_dummies(df, columns=["city"], dtype=int)
    X = encoded.drop(columns=["is_cancelled"])
    y = encoded["is_cancelled"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    return model, X.columns.tolist(), df, acc, auc

model, model_columns, df, accuracy, auc = train_model()

# State
if "result" not in st.session_state:
    st.session_state.result = None


# =====================================================
# 4. HERO SECTION
# =====================================================
st.markdown("""
<div style="margin-bottom: 40px;">
    <div style="color: #6366F1; font-weight: 700; font-size: 0.8rem; letter-spacing: 2px; margin-bottom: 10px;">ENTERPRISE AI SOLUTION</div>
    <h1 class="hero-title">COD Risk Intelligence</h1>
    <p class="hero-subtitle">Next-generation predictive analytics for Pakistani e-commerce. Reduce logistics losses by identifying high-risk orders before they ship.</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 5. TOP METRICS (Business Impact)
# =====================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="glass-card">
        <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 5px;">Total Orders Analyzed</div>
        <div style="font-size: 2rem; font-weight: 800; color: white;">{len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="glass-card">
        <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 5px;">Historical Cancel Rate</div>
        <div style="font-size: 2rem; font-weight: 800; color: #F43F5E;">{df['is_cancelled'].mean()*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="glass-card">
        <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 5px;">Model Accuracy</div>
        <div style="font-size: 2rem; font-weight: 800; color: #10B981;">{accuracy*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="glass-card">
        <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 5px;">ROC-AUC Score</div>
        <div style="font-size: 2rem; font-weight: 800; color: #6366F1;">{auc:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# 6. MAIN DASHBOARD (Tabs)
# =====================================================
tab1, tab2, tab3 = st.tabs(["🔮 Prediction Engine", "📊 Data Insights", "🧠 Model Analytics"])

# --- TAB 1: PREDICTION ---
with tab1:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown('<div class="section-title">⚙️ Order Parameters</div>', unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            city = st.selectbox("Destination City", ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Kahuta", "Faisalabad", "Multan", "Peshawar"])
            order_value = st.number_input("Order Value (PKR)", min_value=500, max_value=500000, value=2500, step=500)
            delivery_days = st.slider("Expected Delivery Days", 1, 10, 3)
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                customer_age_days = st.number_input("Customer Account Age (Days)", min_value=1, max_value=365, value=30)
            with col_b:
                num_items = st.number_input("Total Items in Order", min_value=1, max_value=5, value=1)
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("Run AI Prediction", use_container_width=True)

        if submit:
            input_df = pd.DataFrame([{
                "order_value": order_value,
                "customer_age_days": customer_age_days,
                "num_items": num_items,
                "delivery_days": delivery_days
            }])
            
            for col in model_columns:
                if col.startswith("city_"):
                    input_df[col] = 0
            
            city_col = f"city_{city}"
            if city_col in input_df.columns:
                input_df[city_col] = 1
                
            input_df = input_df[model_columns]
            
            prob = model.predict_proba(input_df)[0][1] * 100
            st.session_state.result = {"prob": prob, "city": city, "value": order_value}

    with right:
        st.markdown('<div class="section-title">🎯 AI Assessment</div>', unsafe_allow_html=True)
        
        if st.session_state.result:
            res = st.session_state.result
            prob = res["prob"]
            
            # Ultra Modern Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                number={'suffix': "%", 'font': {'size': 50, 'color': 'white', 'family': 'Plus Jakarta Sans'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.2)"},
                    'bar': {'color': "#6366F1", 'thickness': 0.3},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.2)"},
                        {'range': [30, 60], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [60, 100], 'color': "rgba(244, 63, 94, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 2},
                        'thickness': 0.75,
                        'value': prob
                    }
                }
            ))
            fig.update_layout(
                height=300,
                margin=dict(t=30, b=10, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "white"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation Card
            if prob >= 60:
                st.error("### 🚨 HIGH RISK DETECTED\n**Action Required:** Call the customer immediately to confirm. Do not dispatch.")
            elif prob >= 30:
                st.warning("### ⚠️ MEDIUM RISK\n**Action Required:** Send an automated WhatsApp confirmation message.")
            else:
                st.success("### ✅ LOW RISK\n**Action Required:** Safe to dispatch. Customer is highly likely to accept delivery.")
                
            # Mini Metrics
            m1, m2 = st.columns(2)
            m1.metric("City", res["city"])
            m2.metric("Order Value", f"PKR {res['value']:,}")
            
        else:
            st.markdown("""
            <div class="glass-card" style="height: 400px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
                <div style="font-size: 4rem; margin-bottom: 20px;">🤖</div>
                <h3 style="color: white; margin-bottom: 10px;">Awaiting Input</h3>
                <p style="color: #94A3B8;">Fill the parameters on the left and run the AI engine to generate a prediction.</p>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 2: DATA INSIGHTS ---
with tab2:
    st.markdown('<div class="section-title">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        fig1 = px.histogram(df, x="delivery_days", color="is_cancelled", barmode="group",
                            title="Delivery Time Impact on Cancellation",
                            color_discrete_sequence=["#10B981", "#F43F5E"])
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig1, use_container_width=True)
        
    with c2:
        city_stats = df.groupby("city")["is_cancelled"].mean().mul(100).reset_index().sort_values("is_cancelled", ascending=False)
        fig2 = px.bar(city_stats, x="city", y="is_cancelled", title="Cancellation Rate by City (%)",
                      color="is_cancelled", color_continuous_scale=["#6366F1", "#F43F5E"])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

# --- TAB 3: MODEL ANALYTICS ---
with tab3:
    st.markdown('<div class="section-title">🧠 Machine Learning Pipeline</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="glass-card" style="margin-bottom: 20px;">
        <h3 style="color: white; margin-bottom: 10px;">Random Forest Classifier</h3>
        <p style="color: #94A3B8;">The AI engine uses an ensemble of 200 decision trees to evaluate the risk profile of an order based on historical patterns.</p>
        <hr style="border-color: rgba(255,255,255,0.1);">
        <div style="display: flex; gap: 40px;">
            <div>
                <div style="color: #94A3B8; font-size: 0.8rem;">ACCURACY</div>
                <div style="color: #10B981; font-size: 1.5rem; font-weight: bold;">{accuracy*100:.2f}%</div>
            </div>
            <div>
                <div style="color: #94A3B8; font-size: 0.8rem;">ROC-AUC</div>
                <div style="color: #6366F1; font-size: 1.5rem; font-weight: bold;">{auc:.3f}</div>
            </div>
            <div>
                <div style="color: #94A3B8; font-size: 0.8rem;">TRAINING SAMPLES</div>
                <div style="color: white; font-size: 1.5rem; font-weight: bold;">{len(df):,}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    importance = pd.DataFrame({"Feature": model_columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
    fig3 = px.bar(importance, x="Importance", y="Feature", orientation="h", title="Feature Importance (What drives cancellations?)",
                  color="Importance", color_continuous_scale=["#6366F1", "#06B6D4"])
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)
