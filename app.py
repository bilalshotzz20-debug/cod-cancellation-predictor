
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="NEXUS AI | COD Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PREMIUM UI DESIGN
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg: #080B14;
    --surface: #101625;
    --surface2: #151E30;
    --border: rgba(148,163,184,.13);
    --cyan: #22D3EE;
    --purple: #A78BFA;
    --text: #F1F5F9;
    --muted: #94A3B8;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(34,211,238,.09), transparent 35%),
        radial-gradient(ellipse at 90% 10%, rgba(139,92,246,.10), transparent 35%),
        var(--bg);
    color: var(--text);
}

#MainMenu, footer, header {
    visibility: hidden;
}

[data-testid="stSidebar"] {
    background: #0C1220;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

.hero {
    padding: 26px 30px;
    border-radius: 20px;
    border: 1px solid rgba(34,211,238,.18);
    background: linear-gradient(120deg,
        rgba(34,211,238,.09),
        rgba(139,92,246,.10),
        rgba(16,22,37,.8));
    margin-bottom: 25px;
}

.hero-title {
    font-size: 36px;
    font-weight: 900;
    letter-spacing: -1.5px;
    background: linear-gradient(90deg,#22D3EE,#A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    color: #94A3B8;
    font-size: 13px;
    margin-top: 8px;
}

.eyebrow {
    color: #22D3EE;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.section-title {
    font-size: 19px;
    font-weight: 750;
    color: #F1F5F9;
    margin: 15px 0;
}

.metric-card {
    background: linear-gradient(145deg,#151E30,#101625);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    min-height: 120px;
}

.metric-label {
    color: #94A3B8;
    font-size: 12px;
    font-weight: 600;
}

.metric-value {
    font-size: 27px;
    font-weight: 800;
    color: #F1F5F9;
    margin-top: 9px;
}

.metric-note {
    color: #22D3EE;
    font-size: 11px;
    margin-top: 5px;
}

.panel {
    background: #101625;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
}

.panel-heading {
    font-size: 16px;
    font-weight: 750;
    color: #F1F5F9;
    margin-bottom: 5px;
}

.panel-caption {
    color: #94A3B8;
    font-size: 12px;
    margin-bottom: 20px;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(100deg,#0891B2,#7C3AED);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: 750;
    min-height: 46px;
    transition: .2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    border: none;
    box-shadow: 0 0 25px rgba(34,211,238,.2);
    transform: translateY(-2px);
}

.stSelectbox div[data-baseweb="select"],
.stNumberInput input {
    background: #0C1220;
    border-radius: 9px;
}

div[data-testid="stMetric"] {
    background: #101625;
    border: 1px solid var(--border);
    padding: 15px;
    border-radius: 13px;
}

div[data-testid="stMetricValue"] {
    color: #22D3EE;
    font-weight: 800;
}

div[data-testid="stTabs"] button {
    font-weight: 700;
}

hr {
    border-color: var(--border);
}

.small-muted {
    font-size: 11px;
    color: #64748B;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL TRAINING
# =====================================================

@st.cache_resource
def train_model():

    np.random.seed(42)

    n = 3000

    cities = [
        "Karachi", "Lahore", "Islamabad",
        "Rawalpindi", "Kahuta", "Faisalabad",
        "Multan", "Peshawar"
    ]

    df = pd.DataFrame({
        "city": np.random.choice(cities, n),
        "order_value": np.random.randint(500, 15001, n),
        "customer_age_days": np.random.randint(1, 366, n),
        "num_items": np.random.randint(1, 6, n),
        "delivery_days": np.random.randint(1, 11, n)
    })

    # Synthetic training labels.
    # These are simulated data, not real customer records.

    logit = (
        -3.0
        + 0.65 * (df["delivery_days"] - 3)
        - 0.003 * np.minimum(df["customer_age_days"], 180)
        + 0.000025 * (df["order_value"] - 2500)
        + 0.12 * (df["num_items"] - 1)
    )

    probability = 1 / (1 + np.exp(-logit))

    df["is_cancelled"] = np.random.binomial(1, probability)

    encoded = pd.get_dummies(
        df,
        columns=["city"],
        dtype=int
    )

    X = encoded.drop(columns=["is_cancelled"])
    y = encoded["is_cancelled"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    return model, list(X.columns), accuracy, auc, df


model, model_columns, accuracy, auc, training_data = train_model()

# =====================================================
# SESSION STATE
# =====================================================

if "result" not in st.session_state:
    st.session_state.result = None

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("""
    <div style="padding:10px 0 25px 0;">
        <div style="font-size:28px;font-weight:900;
        background:linear-gradient(90deg,#22D3EE,#A78BFA);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;">
            ⚡ NEXUS AI
        </div>
        <div style="font-size:10px;color:#64748B;
        letter-spacing:2px;">INTELLIGENCE PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### WORKSPACE")

    page = st.radio(
        "Navigation",
        [
            "Prediction Center",
            "Business Intelligence",
            "Model Performance"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")

    st.markdown("""
    <div style="background:#10251F;padding:12px;
    border-radius:10px;border:1px solid #14532D;">
        <span style="color:#4ADE80;">●</span>
        <b style="color:#4ADE80;"> Model Online</b>
        <p style="font-size:10px;color:#94A3B8;margin:5px 0 0;">
            Random Forest Prediction Engine
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:10px;color:#64748B;">
        NEXUS AI v2.0<br>
        COD RISK INTELLIGENCE<br>
        PAKISTAN EDITION · 2026
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# MAIN HEADER
# =====================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">NEXT-GENERATION E-COMMERCE INTELLIGENCE</div>
    <div class="hero-title">COD Risk Engine</div>
    <div class="hero-sub">
        Predict cancellations. Identify risky orders.
        Make smarter logistics decisions.
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# TOP KPI CARDS
# =====================================================

def metric_card(label, value, note):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


k1, k2, k3, k4 = st.columns(4)

with k1:
    metric_card("TRAINING RECORDS", f"{len(training_data):,}", "Synthetic dataset")

with k2:
    metric_card("MODEL ACCURACY", f"{accuracy*100:.1f}%", "Holdout test set")

with k3:
    metric_card("ROC-AUC", f"{auc:.3f}", "Holdout test set")

with k4:
    metric_card("ALGORITHM", "RF-200", "Random Forest")

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# PAGE 1: PREDICTION CENTER
# =====================================================

if page == "Prediction Center":

    left, right = st.columns([1, 1.15], gap="large")

    with left:

        st.markdown("""
        <div class="panel">
            <div class="eyebrow">ORDER CONFIGURATION</div>
            <div class="panel-heading">📦 Analyze New Order</div>
            <div class="panel-caption">
                Enter customer and delivery information.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("prediction_form"):

            city = st.selectbox(
                "Destination City",
                [
                    "Karachi", "Lahore", "Islamabad",
                    "Rawalpindi", "Kahuta", "Faisalabad",
                    "Multan", "Peshawar"
                ]
            )

            order_value = st.number_input(
                "Order Value (PKR)",
                min_value=500,
                max_value=15000,
                value=2500,
                step=500
            )

            delivery_days = st.slider(
                "Expected Delivery Time",
                1, 10, 3,
                format="%d days"
            )

            customer_age_days = st.number_input(
                "Customer Account Age (Days)",
                min_value=1,
                max_value=365,
                value=30
            )

            num_items = st.number_input(
                "Total Items",
                min_value=1,
                max_value=5,
                value=1
            )

            submitted = st.form_submit_button(
                "⚡ ANALYZE ORDER",
                use_container_width=True
            )

        if submitted:

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

            probability = model.predict_proba(input_df)[0][1] * 100

            # Business decision thresholds
            if probability >= 60:
                risk = "HIGH"
            elif probability >= 30:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            st.session_state.result = {
                "probability": probability,
                "risk": risk,
                "city": city,
                "value": order_value,
                "days": delivery_days
            }

    with right:

        st.markdown("""
        <div class="panel">
            <div class="eyebrow">PREDICTION OUTPUT</div>
            <div class="panel-heading">🧠 AI Risk Assessment</div>
            <div class="panel-caption">
                Your order analysis and recommended action.
            </div>
        </div>
        """, unsafe_allow_html=True)

        result = st.session_state.result

        if result:

            probability = result["probability"]
            risk = result["risk"]

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability,
                number={
                    "suffix": "%",
                    "font": {"size": 42, "color": "#F1F5F9"}
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#94A3B8"
                    },
                    "bar": {"color": "#22D3EE"},
                    "bgcolor": "#101625",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "#123D35"},
                        {"range": [30, 60], "color": "#493C20"},
                        {"range": [60, 100], "color": "#48232E"}
                    ]
                }
            ))

            fig.update_layout(
                height=260,
                margin=dict(t=25, b=10, l=25, r=25),
                paper_bgcolor="#101625",
                font_color="#F1F5F9"
            )

            st.plotly_chart(fig, use_container_width=True)

            if risk == "HIGH":
                st.error("🚨 HIGH RISK ORDER")

                st.markdown("""
                **Recommended actions**
                - Verify customer details before dispatch.
                - Confirm the order through WhatsApp or phone.
                - Review delivery feasibility.
                """)

            elif risk == "MEDIUM":
                st.warning("⚠️ MEDIUM RISK ORDER")

                st.markdown("""
                **Recommended actions**
                - Send an order confirmation message.
                - Monitor delivery progress.
                - Consider proactive customer follow-up.
                """)

            else:
                st.success("✅ LOW RISK ORDER")

                st.markdown("""
                **Recommended actions**
                - Process the order normally.
                - Follow standard dispatch procedures.
                - Monitor delivery as usual.
                """)

            st.divider()

            a, b = st.columns(2)

            a.metric("Destination", result["city"])
            b.metric("Order Value", f"PKR {result['value']:,}")

        else:

            st.markdown("""
            <div style="background:#101625;border:1px dashed
            rgba(34,211,238,.3);border-radius:16px;
            padding:75px 20px;text-align:center;">
                <div style="font-size:48px;">🔮</div>
                <h3 style="color:#E2E8F0;">Awaiting Analysis</h3>
                <p style="color:#64748B;font-size:13px;">
                    Configure your order parameters and
                    run the AI engine to generate a risk report.
                </p>
            </div>
            """, unsafe_allow_html=True)

# =====================================================
# PAGE 2: BUSINESS INTELLIGENCE
# =====================================================

elif page == "Business Intelligence":

    st.markdown("## 📊 Business Intelligence")

    st.caption(
        "Explore patterns in the synthetic training dataset. "
        "These are not actual Pakistani e-commerce statistics."
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            training_data,
            x="delivery_days",
            color="is_cancelled",
            barmode="group",
            title="Orders by Delivery Time",
            labels={
                "delivery_days": "Delivery Days",
                "is_cancelled": "Cancellation"
            },
            color_discrete_sequence=["#22D3EE", "#A78BFA"]
        )

        fig.update_layout(
            paper_bgcolor="#101625",
            plot_bgcolor="#101625",
            font_color="#CBD5E1",
            title_font_color="#F1F5F9"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        city_stats = (
            training_data.groupby("city")["is_cancelled"]
            .mean()
            .mul(100)
            .reset_index(name="cancellation_rate")
            .sort_values("cancellation_rate", ascending=False)
        )

        fig = px.bar(
            city_stats,
            x="city",
            y="cancellation_rate",
            title="Simulated Cancellation Rate by City",
            color="cancellation_rate",
            color_continuous_scale=["#22D3EE", "#A78BFA"]
        )

        fig.update_layout(
            paper_bgcolor="#101625",
            plot_bgcolor="#101625",
            font_color="#CBD5E1",
            title_font_color="#F1F5F9",
            coloraxis_showscale=False,
            xaxis_title="City",
            yaxis_title="Cancellation Rate (%)"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💰 Order Value Distribution")

    fig = px.histogram(
        training_data,
        x="order_value",
        nbins=30,
        title="Simulated Order Value Distribution",
        color_discrete_sequence=["#22D3EE"]
    )

    fig.update_layout(
        paper_bgcolor="#101625",
        plot_bgcolor="#101625",
        font_color="#CBD5E1",
        title_font_color="#F1F5F9"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Recent Training Records")

    st.dataframe(
        training_data.head(20),
        use_container_width=True,
        hide_index=True
    )

# =====================================================
# PAGE 3: MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.markdown("## 🧠 Model Performance")

    st.caption("Evaluation using a held-out synthetic test dataset.")

    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", f"{accuracy*100:.2f}%")
    c2.metric("ROC-AUC", f"{auc:.3f}")
    c3.metric("Trees", "200")

    st.markdown("### Feature Importance")

    importance = pd.DataFrame({
        "Feature": model_columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance", ascending=True
    )

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance",
        color="Importance",
        color_continuous_scale=["#22D3EE", "#A78BFA"]
    )

    fig.update_layout(
        paper_bgcolor="#101625",
        plot_bgcolor="#101625",
        font_color="#CBD5E1",
        title_font_color="#F1F5F9",
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### About the AI Engine")

    st.markdown("""
    <div class="panel">
        <div class="panel-heading">Random Forest Classifier</div>
        <p style="color:#94A3B8;font-size:13px;">
        The model combines 200 decision trees to estimate
        the likelihood of a COD order cancellation.
        </p>
        <hr>
        <b>Input Features</b>
        <ul style="color:#CBD5E1;">
            <li>Destination City</li>
            <li>Order Value</li>
            <li>Customer Account Age</li>
            <li>Number of Items</li>
            <li>Expected Delivery Days</li>
        </ul>
        <p style="color:#FBBF24;font-size:12px;">
        Important: The current model uses simulated data.
        Its predictions are experimental and should not be
        used for real dispatch decisions until validated
        against actual historical order records.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown("""
<div style="text-align:center;padding:12px;color:#64748B;
font-size:11px;">
    ⚡ NEXUS AI · COD RISK INTELLIGENCE · 2026<br>
    Built for smarter e-commerce logistics.
</div>
""", unsafe_allow_html=True)
