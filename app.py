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
# ULTRA PREMIUM UI
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg: #070B13;
    --surface: #0D1422;
    --surface2: #121D30;
    --border: rgba(148,163,184,.14);
    --cyan: #67E8F9;
    --purple: #A78BFA;
    --text: #F1F5F9;
    --muted: #94A3B8;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(900px 500px at 5% -10%, rgba(14,165,233,.13), transparent 60%),
        radial-gradient(800px 480px at 100% 0%, rgba(124,58,237,.14), transparent 58%),
        linear-gradient(180deg,#070B13 0%,#090F1B 55%,#070B13 100%);
    color: var(--text);
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    max-width: 1600px;
    padding: 2rem 2.4rem 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0B1220 0%,#080D17 100%);
    border-right: 1px solid rgba(148,163,184,.12);
}

[data-testid="stSidebar"] > div {
    padding-top: 1.4rem;
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 9px 12px;
    border-radius: 11px;
    transition: all .2s ease;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(103,232,249,.08);
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 7px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    font-weight: 650;
    font-size: 13px;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 30px 34px;
    border-radius: 23px;
    border: 1px solid rgba(103,232,249,.19);
    background: linear-gradient(
        115deg,
        rgba(14,165,233,.13),
        rgba(124,58,237,.12) 58%,
        rgba(15,23,42,.76)
    );
    box-shadow: 0 18px 60px rgba(0,0,0,.22);
    margin-bottom: 25px;
}

.hero:after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -85px;
    top: -135px;
    border-radius: 50%;
    border: 1px solid rgba(103,232,249,.15);
    box-shadow:
        0 0 0 28px rgba(103,232,249,.025),
        0 0 0 58px rgba(167,139,250,.025);
}

.hero-title {
    font-size: clamp(30px,4vw,43px);
    font-weight: 900;
    line-height: 1.08;
    letter-spacing: -1.8px;
    background: linear-gradient(90deg,#22D3EE,#A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    max-width: 680px;
    font-size: 14px;
    line-height: 1.7;
    color: #A8B6CA;
    margin-top: 8px;
}

.eyebrow {
    color: #67E8F9;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2.4px;
    margin-bottom: 8px;
}

.section-title {
    font-size: 19px;
    font-weight: 750;
    color: #F1F5F9;
    margin: 15px 0;
}

.metric-card {
    position: relative;
    overflow: hidden;
    min-height: 132px;
    padding: 21px 22px;
    background: linear-gradient(145deg,rgba(20,32,52,.96),rgba(12,20,34,.96));
    border: 1px solid rgba(148,163,184,.14);
    border-radius: 17px;
    box-shadow: 0 10px 28px rgba(0,0,0,.16);
    transition: transform .2s,border-color .2s;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(103,232,249,.32);
}

.metric-card:after {
    content: "";
    position: absolute;
    right: -28px;
    top: -38px;
    width: 105px;
    height: 105px;
    border-radius: 50%;
    background: rgba(103,232,249,.055);
}

.metric-label {
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    font-size: 10px;
    font-weight: 600;
}

.metric-value {
    font-size: 29px;
    font-weight: 800;
    color: #F1F5F9;
    margin-top: 9px;
    letter-spacing: -.8px;
}

.metric-note {
    color: #8EA2BC;
    font-size: 11px;
    margin-top: 5px;
}

.panel {
    background: linear-gradient(145deg,rgba(17,27,45,.96),rgba(11,18,31,.98));
    border: 1px solid rgba(148,163,184,.14);
    border-radius: 19px;
    padding: 23px;
    margin-bottom: 18px;
    box-shadow: 0 12px 34px rgba(0,0,0,.14);
}

.panel-heading {
    font-size: 17px;
    font-weight: 750;
    color: #F1F5F9;
    margin-bottom: 5px;
}

.panel-caption {
    color: #8FA1B8;
    font-size: 12px;
    line-height: 1.6;
    margin-bottom: 20px;
}

.stButton > button,
.stFormSubmitButton > button {
    min-height: 49px;
    border-radius: 12px;
    letter-spacing: .35px;
    background: linear-gradient(
        105deg,
        #0891B2 0%,
        #6366F1 55%,
        #7C3AED 100%
    );
    color: white;
    border: none;
    font-weight: 750;
    box-shadow: 0 8px 24px rgba(34,211,238,.12);
    transition: .2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    box-shadow: 0 10px 30px rgba(99,102,241,.24);
    transform: translateY(-2px);
}

.stTextInput input,
.stNumberInput input,
.stSelectbox [data-baseweb="select"],
.stMultiSelect [data-baseweb="select"] {
    background: #0A1220 !important;
    border-color: rgba(148,163,184,.2) !important;
    border-radius: 11px !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] {
    padding-top: 8px;
}

[data-testid="stTabs"] button {
    padding: 11px 15px;
    font-weight: 700;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 13px;
    overflow: hidden;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg,#121D30,#0D1422);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 15px 17px;
}

[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

[data-testid="stMetricValue"] {
    color: #67E8F9;
    font-weight: 800;
    letter-spacing: -.6px;
}

.stAlert {
    border-radius: 13px;
    border-width: 1px;
}

hr {
    margin: 1.5rem 0;
    border-color: rgba(148,163,184,.13);
}

.small-muted {
    font-size: 11px;
    color: #64748B;
}

@media (max-width: 900px) {
    .block-container {
        padding: 1.2rem 1rem 2rem;
    }

    .hero {
        padding: 24px 21px;
    }

    .panel {
        padding: 17px;
    }
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
        "Karachi",
        "Lahore",
        "Islamabad",
        "Rawalpindi",
        "Kahuta",
        "Faisalabad",
        "Multan",
        "Peshawar"
    ]

    df = pd.DataFrame({
        "city": np.random.choice(cities, n),
        "order_value": np.random.randint(500, 15001, n),
        "customer_age_days": np.random.randint(1, 366, n),
        "num_items": np.random.randint(1, 6, n),
        "delivery_days": np.random.randint(1, 11, n)
    })

    # Synthetic training labels
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
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    return model, X.columns.tolist(), df, accuracy, auc


model, model_columns, training_data, accuracy, auc = train_model()

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
    <div style="padding:8px 8px 25px;">
        <div style="font-size:27px;font-weight:900;letter-spacing:-1px;">
            <span style="color:#67E8F9;">NEXUS</span>
            <span style="color:#A78BFA;">AI</span>
        </div>
        <div style="font-size:10px;letter-spacing:2px;color:#64748B;">
            COD INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="eyebrow">WORKSPACE</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Prediction Dashboard",
            "Business Intelligence",
            "Model Performance"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        '<div class="eyebrow">SYSTEM STATUS</div>',
        unsafe_allow_html=True
    )

    st.success("AI Engine Online")

    st.markdown("""
    <div style="
        background:#101A2B;
        border:1px solid rgba(148,163,184,.13);
        border-radius:12px;
        padding:14px;
        margin-top:12px;
    ">
        <div style="color:#94A3B8;font-size:11px;">MODEL VERSION</div>
        <div style="color:#F1F5F9;font-size:15px;font-weight:750;">
            Random Forest v2.1
        </div>
        <div style="color:#94A3B8;font-size:11px;margin-top:8px;">
            Training records
        </div>
        <div style="color:#67E8F9;font-size:20px;font-weight:800;">
            3,000
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        position:fixed;
        bottom:20px;
        color:#64748B;
        font-size:10px;
    ">
        NEXUS AI · VERSION 2.1<br>
        BUILT FOR SMARTER LOGISTICS
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# MAIN HERO
# =====================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">INTELLIGENT E-COMMERCE ANALYTICS</div>
    <div class="hero-title">COD Risk Intelligence</div>
    <div class="hero-sub">
        Predict potential Cash-on-Delivery cancellations,
        understand order patterns and make smarter logistics
        decisions with AI-powered insights.
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# PAGE 1: PREDICTION DASHBOARD
# =====================================================

if page == "Prediction Dashboard":

    total_orders = len(training_data)
    cancellation_rate = training_data["is_cancelled"].mean() * 100
    avg_order_value = training_data["order_value"].mean()

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        ("TOTAL ORDERS", f"{total_orders:,}", "Synthetic dataset"),
        ("CANCELLATION RATE", f"{cancellation_rate:.1f}%", "Simulated data"),
        ("AVG. ORDER VALUE", f"PKR {avg_order_value:,.0f}", "Average value"),
        ("MODEL ACCURACY", f"{accuracy*100:.1f}%", "Held-out test set")
    ]

    for column, (label, value, note) in zip(
        [col1, col2, col3, col4],
        metrics
    ):
        with column:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1.1], gap="large")

    with left:

        st.markdown("""
        <div class="panel">
            <div class="eyebrow">ORDER CONFIGURATION</div>
            <div class="panel-heading">Analyze a New Order</div>
            <div class="panel-caption">
                Enter order details to estimate cancellation risk.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("prediction_form"):

            city = st.selectbox(
                "Delivery Destination",
                [
                    "Karachi",
                    "Lahore",
                    "Islamabad",
                    "Rawalpindi",
                    "Kahuta",
                    "Faisalabad",
                    "Multan",
                    "Peshawar"
                ]
            )

            order_value = st.number_input(
                "Order Value (PKR)",
                min_value=100,
                max_value=500000,
                value=2500,
                step=100
            )

            delivery_days = st.slider(
                "Estimated Delivery Time",
                min_value=1,
                max_value=10,
                value=3,
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
                    "font": {
                        "size": 42,
                        "color": "#F1F5F9"
                    }
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
            <div style="
                background:#101625;
                border:1px dashed rgba(34,211,238,.3);
                border-radius:16px;
                padding:75px 20px;
                text-align:center;
            ">
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

    st.caption(
        "Evaluation using a held-out synthetic test dataset."
    )

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
        "Importance",
        ascending=True
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
<div style="
    text-align:center;
    padding:12px;
    color:#64748B;
    font-size:11px;
">
    ⚡ NEXUS AI · COD RISK INTELLIGENCE · 2026<br>
    Built for smarter e-commerce logistics.
</div>
""", unsafe_allow_html=True)
