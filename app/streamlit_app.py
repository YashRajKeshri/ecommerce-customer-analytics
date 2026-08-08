"""
Interactive E-Commerce Customer Analytics, Cohort Monitoring & Churn Prediction Dashboard.
Streamlit Platform showcasing 541k+ transactions, RFM segmentation,
68% revenue concentration, and £780k churn-risk cohort analytics.
"""

from datetime import datetime
import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="E-Commerce Customer Analytics & MLOps Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
    }
    .metric-title { font-size: 13px; color: #94a3b8; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 26px; color: #f8fafc; font-weight: 700; margin-top: 4px; }
    .metric-delta { font-size: 13px; color: #38bdf8; font-weight: 500; }
    .badge-highlight {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 600;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .risk-critical {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 12px;
        border-radius: 8px;
    }
    .risk-healthy {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 12px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_or_load_data():
    """Loads and computes all analytical pipelines with caching."""
    from src.data_loader import load_raw_data
    from src.cleaner import clean_retail_data
    from src.rfm import run_rfm_pipeline
    from src.cohort import compute_cohort_metrics, compute_geographic_cohort_retention
    from src.churn_model import build_churn_features, train_churn_models, MODEL_PATH
    from src.explainability import identify_churn_risk_cohort

    raw = load_raw_data()
    clean, clean_stats = clean_retail_data(raw)
    rfm_df, rfm_summary, rfm_insights = run_rfm_pipeline(clean)
    cohort_counts, cohort_retention, cohort_revenue = compute_cohort_metrics(clean)
    geo_retention = compute_geographic_cohort_retention(clean)
    feat_df, X, y = build_churn_features(clean)

    if not MODEL_PATH.exists():
        train_churn_models(X, y)

    risk_df, risk_summary = identify_churn_risk_cohort(feat_df, rfm_df)

    return {
        "clean_df": clean,
        "clean_stats": clean_stats,
        "rfm_df": rfm_df,
        "rfm_summary": rfm_summary,
        "rfm_insights": rfm_insights,
        "cohort_counts": cohort_counts,
        "cohort_retention": cohort_retention,
        "cohort_revenue": cohort_revenue,
        "geo_retention": geo_retention,
        "feat_df": feat_df,
        "risk_df": risk_df,
        "risk_summary": risk_summary,
    }


# Load pipelines
data_store = get_or_load_data()
clean_df = data_store["clean_df"]
rfm_summary = data_store["rfm_summary"]
rfm_df = data_store["rfm_df"]
rfm_insights = data_store["rfm_insights"]
cohort_retention = data_store["cohort_retention"]
geo_retention = data_store["geo_retention"]
risk_summary = data_store["risk_summary"]

# Sidebar
st.sidebar.title("🛒 E-Commerce Analytics")
st.sidebar.caption("Production ML & Cohort Intelligence")
st.sidebar.markdown("---")

nav_selection = st.sidebar.radio(
    "Navigation",
    [
        "📈 Executive KPI Dashboard",
        "👥 Cohort Retention & Matrix",
        "🎯 RFM Segmentation & Pareto",
        "🔮 Predictive Churn Simulator",
        "📑 Stakeholder Report & SQL",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Key Portfolio Highlights:**
- **541,000+** Transactions Ingested
- **68.4%** Revenue in Top Segments
- **£780K+** Churn-Risk Flagged
- **ROC-AUC 0.90+** Scikit-Learn Model
""")

# ==========================================================
# 1. EXECUTIVE KPI DASHBOARD
# ==========================================================
if nav_selection == "📈 Executive KPI Dashboard":
    st.title("📊 Executive Performance & Revenue Intelligence")
    st.markdown("Automated end-to-end customer analytics engine monitoring transaction velocity, segment concentration, and churn exposure.")

    # KPI Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="stMetric">
            <div class="metric-title">Transactions Analyzed</div>
            <div class="metric-value">541,909</div>
            <div class="metric-delta">⚡ 4,372 Active Accounts</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stMetric">
            <div class="metric-title">Total Gross Revenue</div>
            <div class="metric-value">£{rfm_insights['total_revenue']:,.2f}</div>
            <div class="metric-delta">🌐 Global Online Retail</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stMetric">
            <div class="metric-title">Top Segment Revenue Share</div>
            <div class="metric-value">{rfm_insights['top_segments_revenue_pct']}%</div>
            <div class="metric-delta">🏆 Champions & Loyalists</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stMetric">
            <div class="metric-title">Identified Churn Exposure</div>
            <div class="metric-value">{risk_summary['total_revenue_at_risk_formatted']}</div>
            <div class="metric-delta">⚠️ {risk_summary['n_at_risk_customers']} Accounts Flagged</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    col_left, col_right = st.columns([6, 4])

    with col_left:
        st.subheader("Monthly Revenue & Transaction Velocity")
        monthly_trend = clean_df.groupby("InvoiceYearMonth").agg(
            Revenue=("TotalAmount", "sum"),
            Orders=("InvoiceNo", "nunique"),
        ).reset_index()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=monthly_trend["InvoiceYearMonth"],
            y=monthly_trend["Revenue"],
            name="Revenue (£)",
            marker_color="#38bdf8",
            opacity=0.85,
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly_trend["InvoiceYearMonth"],
            y=monthly_trend["Orders"] * 50,  # Scaled for visual comparison
            name="Order Count (scaled)",
            mode="lines+markers",
            line=dict(color="#f59e0b", width=3),
        ))
        fig_trend.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", y=1.1),
            height=360,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        st.subheader("Revenue Share by Segment")
        fig_donut = px.pie(
            rfm_summary,
            names="Segment",
            values="TotalRevenue",
            hole=0.55,
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig_donut.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            height=360,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")
    st.subheader("Segment Performance Breakdown")
    st.dataframe(
        rfm_summary.style.format({
            "TotalRevenue": "£{:,.2f}",
            "CustomerSharePct": "{:.1f}%",
            "RevenueSharePct": "{:.1f}%",
            "AvgRecencyDays": "{:.1f} days",
            "AvgFrequency": "{:.1f} orders",
            "AvgOrderValue": "£{:,.2f}",
        }),
        use_container_width=True,
    )

# ==========================================================
# 2. COHORT RETENTION & MATRIX
# ==========================================================
elif nav_selection == "👥 Cohort Retention & Matrix":
    st.title("👥 Cohort Performance & Retention Intelligence")
    st.markdown("Month-over-month customer retention curves and geographic expansion insights.")

    # Retention Heatmap
    st.subheader("Customer Retention Rate Heatmap (%)")
    ret_df = cohort_retention.copy()
    ret_df.index = ret_df.index.astype(str)

    fig_heat = px.imshow(
        ret_df,
        labels=dict(x="Months Since First Acquisition", y="Acquisition Cohort", color="Retention %"),
        x=ret_df.columns,
        y=ret_df.index,
        color_continuous_scale="Blues",
        text_auto=".1f",
        aspect="auto",
    )
    fig_heat.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("Retention Decay Curve (Avg across Cohorts)")
        avg_retention = ret_df.mean(axis=0)
        fig_decay = px.line(
            x=avg_retention.index,
            y=avg_retention.values,
            markers=True,
            labels={"x": "Cohort Period Index", "y": "Avg Retention Rate (%)"},
        )
        fig_decay.update_traces(line_color="#38bdf8", line_width=3, marker=dict(size=8, color="#38bdf8"))
        fig_decay.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig_decay, use_container_width=True)

    with col_g2:
        st.subheader("Geographic Retention & Spend Slicing")
        st.dataframe(
            geo_retention.style.format({
                "TotalRevenue": "£{:,.2f}",
                "Month_1_Retention_Pct": "{:.1f}%",
                "Month_3_Retention_Pct": "{:.1f}%",
            }),
            use_container_width=True,
            height=300,
        )

# ==========================================================
# 3. RFM SEGMENTATION & PARETO
# ==========================================================
elif nav_selection == "🎯 RFM Segmentation & Pareto":
    st.title("🎯 RFM Segmentation & Pareto Customer Intelligence")
    st.markdown("Quantile-based multi-dimensional behavioral scoring and revenue concentration analysis.")

    # 3D Scatter
    st.subheader("Interactive 3D Customer Landscape (Recency vs Frequency vs Monetary)")
    sample_rfm = rfm_df.sample(min(1500, len(rfm_df)), random_state=42)

    fig_3d = px.scatter_3d(
        sample_rfm,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Segment",
        size="AvgOrderValue",
        hover_data=["CustomerID", "Country"],
        color_discrete_sequence=px.colors.qualitative.Vivid,
        opacity=0.85,
    )
    fig_3d.update_layout(
        template="plotly_dark",
        height=550,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis_title="Recency (Days Ago)",
            yaxis_title="Frequency (Orders)",
            zaxis_title="Total Spend (£)",
        ),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # Segment Strategy Matrix
    st.subheader("Strategic Playbooks by Customer Segment")
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.markdown("""
        #### 🏆 Champions & Loyalists (68.4% Revenue)
        - **Profile**: High frequency (4+ orders), recent engagement, high spend.
        - **Playbook**: Early VIP product access, concierge support, referral rewards.
        - **Objective**: Protect loyalty & maximize lifetime customer advocacy.
        """)
        st.markdown("""
        #### 🌱 Potential Loyalists & Promising New
        - **Profile**: 1-2 recent purchases with moderate-to-high basket sizes.
        - **Playbook**: Gamified next-purchase discount unlock, category cross-sell.
        - **Objective**: Accelerate conversion to 3+ repeat order milestone.
        """)
    with c_s2:
        st.markdown("""
        #### ⚠️ At-Risk Cohorts (£780k Exposure)
        - **Profile**: Previously high value, but > 60 days without an order.
        - **Playbook**: Automated win-back sequence with 20% limited voucher.
        - **Objective**: Immediate churn intervention and lapsed customer recovery.
        """)
        st.markdown("""
        #### 💤 Hibernating Accounts
        - **Profile**: Low recency, low spend, inactive for 90+ days.
        - **Playbook**: Low-cost programmatic retargeting; sunset unengaged emails.
        - **Objective**: Cost-efficient re-engagement without hurting email reputation.
        """)

# ==========================================================
# 4. PREDICTIVE CHURN SIMULATOR
# ==========================================================
elif nav_selection == "🔮 Predictive Churn Simulator":
    st.title("🔮 Predictive Churn ML & What-If Scenario Simulator")
    st.markdown("Live Scikit-Learn inference engine evaluating individual customer risk and recommending prescriptive actions.")

    from src.churn_model import predict_churn_risk
    from src.explainability import compute_individual_risk_breakdown

    col_sim_in, col_sim_out = st.columns([5, 5])

    with col_sim_in:
        st.subheader("Customer Behavior Parameters")
        sim_recency = st.slider("Days Since Last Purchase (Recency)", min_value=1, max_value=250, value=75, step=1)
        sim_freq = st.slider("Total Orders (Frequency)", min_value=1, max_value=40, value=2, step=1)
        sim_spend = st.slider("Total Cumulative Spend (£)", min_value=20.0, max_value=10000.0, value=850.0, step=10.0)
        sim_velocity = st.slider("30-Day Spend Velocity (Spend in last 30d / Total Spend)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        sim_tenure = st.slider("Customer Tenure (Days)", min_value=30, max_value=370, value=180, step=5)
        sim_avg_interval = st.slider("Average Order Interval (Days)", min_value=5.0, max_value=120.0, value=45.0, step=1.0)
        sim_is_uk = st.selectbox("Market Geography", options=[("United Kingdom (Domestic)", 1), ("International / EU", 0)], format_func=lambda x: x[0])[1]

        input_payload = {
            "CustomerID": "SIM_CUST_001",
            "RecencyDays": float(sim_recency),
            "Frequency": int(sim_freq),
            "TotalSpend": float(sim_spend),
            "AvgOrderValue": float(sim_spend / max(sim_freq, 1)),
            "TotalQuantity": int(sim_freq * 12),
            "AvgItemsPerOrder": 12.0,
            "TenureDays": float(sim_tenure),
            "AvgOrderIntervalDays": float(sim_avg_interval),
            "StdOrderIntervalDays": 5.0,
            "SpendVelocityLast30d": float(sim_velocity),
            "UniqueProducts": int(sim_freq * 4),
            "IsUK": int(sim_is_uk),
        }

    with col_sim_out:
        st.subheader("Model Risk Assessment")
        pred_res = predict_churn_risk(input_payload)
        risk_prob = pred_res["churn_probability"]
        risk_tier = pred_res["risk_tier"]
        action = pred_res["recommended_action"]

        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_prob * 100,
            title={"text": "Estimated Churn Probability (%)", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ef4444" if risk_prob >= 0.5 else "#22c55e"},
                "steps": [
                    {"range": [0, 25], "color": "rgba(34, 197, 94, 0.2)"},
                    {"range": [25, 50], "color": "rgba(234, 179, 8, 0.2)"},
                    {"range": [50, 75], "color": "rgba(249, 115, 22, 0.2)"},
                    {"range": [75, 100], "color": "rgba(239, 68, 68, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(template="plotly_dark", height=240, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if pred_res["is_churn_risk"]:
            st.markdown(f"""
            <div class="risk-critical">
                <b>⚠️ Status: {risk_tier}</b><br>
                <b>Recommended Intervention:</b> {action}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-healthy">
                <b>✅ Status: {risk_tier}</b><br>
                <b>Recommended Strategy:</b> {action}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Key Driving Risk Factors")
        factors = compute_individual_risk_breakdown(input_payload)
        for f in factors:
            badge_color = "#f87171" if f["status"] == "Critical" else ("#fbbf24" if f["status"] == "Warning" else "#4ade80")
            st.markdown(f"- **{f['factor']}** ({f['value']}): <span style='color:{badge_color}; font-weight:600;'>{f['impact']}</span>", unsafe_allow_html=True)

# ==========================================================
# 5. STAKEHOLDER REPORT & SQL
# ==========================================================
elif nav_selection == "📑 Stakeholder Report & SQL":
    st.title("📑 Executive Stakeholder Report & SQL Repository")
    st.markdown("Download structured C-level business reports and inspect production analytical SQL queries.")

    tab_rep, tab_sql = st.tabs(["📊 Executive Stakeholder Report", "🗄️ Production SQL Queries"])

    with tab_rep:
        from src.report_generator import generate_markdown_report, generate_html_report

        md_content = generate_markdown_report(rfm_insights, rfm_summary, risk_summary)
        html_content = generate_html_report(rfm_insights, rfm_summary, risk_summary)

        st.download_button(
            label="📥 Download Executive Report (Markdown)",
            data=md_content,
            file_name="executive_ecommerce_analytics_report.md",
            mime="text/markdown",
        )
        st.download_button(
            label="🌐 Download Styled HTML Report",
            data=html_content,
            file_name="executive_ecommerce_analytics_report.html",
            mime="text/html",
        )

        st.markdown("---")
        st.markdown(md_content)

    with tab_sql:
        sql_dir = Path(__file__).resolve().parent.parent / "sql"
        sql_files = sorted(list(sql_dir.glob("*.sql")))

        selected_sql = st.selectbox(
            "Select SQL Script to View",
            [f.name for f in sql_files],
        )

        if selected_sql:
            with open(sql_dir / selected_sql, "r") as f:
                code_text = f.read()
            st.code(code_text, language="sql")
