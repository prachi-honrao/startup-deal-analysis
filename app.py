"""
Startup Funding & Deal Outcome Analysis
Streamlit dashboard: Home | Univariate | Bivariate | Multivariate
Run:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Page config & style
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Startup Funding & Deal Outcome Analysis",
    page_icon="📊",
    layout="wide",
)

PRIMARY = "#3B82C4"
DEAL_COLORS = {"Yes": "#2E8B57", "No": "#C0504D"}
PROFIT_COLORS = {"Profit": "#2E8B57", "Loss": "#C0504D"}

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem;}
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.12);
        border: 1px solid rgba(128, 128, 128, 0.35);
        padding: 12px 16px; border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = Path(__file__).parent / "data" / "shark_tank_corrected.csv"

COLUMN_INFO = pd.DataFrame(
    [
        ("startup_id", "Identifier", "Unique ID of the pitch (excluded from analysis)"),
        ("startup_name", "Text", "Name of the startup (excluded from analysis)"),
        ("industry", "Categorical", "Industry / sector of the startup"),
        ("city", "Categorical", "City the startup is based in"),
        ("season", "Discrete", "Season in which the startup pitched (1-8)"),
        ("business_stage", "Categorical", "Idea, Early Stage, Growth Stage or Established"),
        ("sales_channel", "Categorical", "Online, Offline or Both"),
        ("founder_count", "Discrete", "Number of founders (1-5)"),
        ("annual_sales_lakh_inr", "Numeric", "Annual sales in ₹ Lakh"),
        ("profit_margin_pct", "Numeric", "Profit margin in % (negative = loss-making)"),
        ("profit_status", "Categorical", "Profit or Loss"),
        ("asking_amount_lakh_inr", "Numeric", "Funding asked by the founder in ₹ Lakh"),
        ("equity_offered_pct", "Numeric", "Equity offered in exchange, in %"),
        ("deal_made", "Target", "Whether a deal was closed (Yes / No)"),
        ("deal_amount_lakh_inr", "Numeric", "Deal amount in ₹ Lakh (0 when no deal)"),
    ],
    columns=["Column", "Type", "Description"],
)

LABELS = {
    "industry": "Industry",
    "city": "City",
    "season": "Season",
    "business_stage": "Business Stage",
    "sales_channel": "Sales Channel",
    "founder_count": "Founder Count",
    "profit_status": "Profit Status",
    "deal_made": "Deal Made",
    "annual_sales_lakh_inr": "Annual Sales (₹ Lakh)",
    "profit_margin_pct": "Profit Margin (%)",
    "asking_amount_lakh_inr": "Asking Amount (₹ Lakh)",
    "equity_offered_pct": "Equity Offered (%)",
    "deal_amount_lakh_inr": "Deal Amount (₹ Lakh)",
    "deal_rate": "Deal Rate (%)",
}
CAT_COLS = ["industry", "city", "season", "business_stage", "sales_channel",
            "founder_count", "profit_status"]
NUM_COLS = ["annual_sales_lakh_inr", "profit_margin_pct", "asking_amount_lakh_inr",
            "equity_offered_pct", "deal_amount_lakh_inr"]
STAGE_ORDER = ["Idea", "Early Stage", "Growth Stage", "Established"]


def lab(c: str) -> str:
    return LABELS.get(c, c)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()
    df = df.drop_duplicates().reset_index(drop=True)
    for c in ["industry", "sales_channel"]:  # small share of missing categories
        df[c] = df[c].fillna("Unknown")
    df["deal_flag"] = (df["deal_made"] == "Yes").astype(int)
    return df


def deal_rate_table(d: pd.DataFrame, col: str) -> pd.DataFrame:
    g = d.groupby(col, observed=True).agg(pitches=("deal_flag", "size"),
                                          deals=("deal_flag", "sum")).reset_index()
    g["deal_rate"] = (g["deals"] / g["pitches"] * 100).round(2)
    return g


def fmt_lakh(x: float) -> str:
    """Format a ₹ Lakh value in Lakh / Crore for readability."""
    if pd.isna(x):
        return "-"
    if abs(x) >= 100:
        return f"₹{x / 100:,.1f} Cr"
    return f"₹{x:,.1f} L"


def style(fig, height=420):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10),
                      legend_title_text="")
    return fig


def clip_for_view(s: pd.Series):
    """Cap the extreme right tail (as the notebook does for sales) so the bulk of the data is visible."""
    s = s.dropna()
    if s.skew() > 1:
        cap = s.quantile(0.99)
        return s[s <= cap], f"Chart view limited to values up to the 99th percentile ({cap:,.1f}); the summary table below uses all values."
    return s, ""


def histogram_fig(s: pd.Series, bins: int, title: str, label: str):
    counts, edges = np.histogram(s, bins=bins)
    centers, width = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
    hover = [f"{a:,.2f} – {b:,.2f}<br>Count: {c:,}" for a, b, c in zip(edges[:-1], edges[1:], counts)]
    fig = go.Figure(go.Bar(x=centers, y=counts, width=width, marker_color=PRIMARY,
                           hovertext=hover, hoverinfo="text"))
    fig.update_layout(title=title, bargap=0.03, xaxis_title=label, yaxis_title="Number of Startups")
    return fig


df = load_data()

# ----------------------------------------------------------------------------
# Sidebar: navigation + global filters
# ----------------------------------------------------------------------------
st.sidebar.title("📊 Startup Funding")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📈 Univariate", "🔗 Bivariate", "🧩 Multivariate"])

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

seasons = st.sidebar.multiselect("Season", sorted(df["season"].unique()),
                                 default=sorted(df["season"].unique()))
industries = st.sidebar.multiselect("Industry", sorted(df["industry"].unique()),
                                    placeholder="All industries")
cities = st.sidebar.multiselect("City", sorted(df["city"].unique()),
                                placeholder="All cities")
stages = st.sidebar.multiselect("Business Stage", STAGE_ORDER, placeholder="All stages")
profit_sel = st.sidebar.radio("Profit / Loss", ["All", "Profit", "Loss"], horizontal=True)
channels = st.sidebar.multiselect("Sales Channel", sorted(df["sales_channel"].unique()),
                                  placeholder="All channels")

f = df[df["season"].isin(seasons)]
if industries:
    f = f[f["industry"].isin(industries)]
if cities:
    f = f[f["city"].isin(cities)]
if stages:
    f = f[f["business_stage"].isin(stages)]
if profit_sel != "All":
    f = f[f["profit_status"] == profit_sel]
if channels:
    f = f[f["sales_channel"].isin(channels)]

st.sidebar.caption(f"Showing **{len(f):,}** of {len(df):,} pitches")

if f.empty:
    st.title("Startup Funding & Deal Outcome Analysis")
    st.warning("No data matches the selected filters. Please relax the filters in the sidebar.")
    st.stop()


def kpi_row(d: pd.DataFrame):
    funded = d[d["deal_flag"] == 1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Pitches", f"{len(d):,}")
    c2.metric("Deals Closed", f"{int(d['deal_flag'].sum()):,}")
    c3.metric("Deal Rate", f"{d['deal_flag'].mean() * 100:.1f}%")
    c4.metric("Median Ask", fmt_lakh(d["asking_amount_lakh_inr"].median()))
    c5.metric("Total Funded", fmt_lakh(funded["deal_amount_lakh_inr"].sum()))


# ----------------------------------------------------------------------------
# HOME
# ----------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("📊 Startup Funding & Deal Outcome Analysis")
    st.caption("Exploratory data analysis of startup pitches, funding asks and deal outcomes")

    kpi_row(f)
    st.markdown("")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("🎯 Project Objective")
        st.markdown(
            """
            Understand **what separates startups that close a funding deal from those that don't**.
            The analysis studies pitch patterns across industries, cities, business stages and
            profitability, and looks at how the funding ask, equity offered and sales relate to the
            final deal outcome.

            **Key questions**
            - What share of pitches end in a deal, and how has it moved across seasons?
            - Do **profit-making** startups close deals more often than **loss-making** ones?
            - Do industry, city or business stage change a startup's chances?
            - How are asking amount, equity offered and annual sales related to each other?
            """
        )
    with right:
        st.subheader("🗂️ Dataset at a Glance")
        st.markdown(
            f"""
            | | |
            |---|---|
            | Rows (after de-duplication) | **{len(df):,}** |
            | Columns | **{df.shape[1] - 1}** |
            | Seasons covered | **{df['season'].min()} – {df['season'].max()}** |
            | Industries | **{df['industry'].nunique()}** |
            | Cities | **{df['city'].nunique()}** |
            | Target variable | `deal_made` (Yes / No) |
            | Source file | `shark_tank_corrected.csv` |
            """
        )

    st.subheader("📋 Column Information")
    st.dataframe(COLUMN_INFO, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🧹 Data Preparation")
        st.markdown(
            """
            - Column names standardised to lowercase, underscore-separated.
            - **450 duplicate rows** removed.
            - Missing values (all under 2%) in `industry` and `sales_channel` are labelled
              *Unknown*; missing numeric values are left out of the relevant charts.
            - Extreme values in sales, asking and deal amounts are **kept** – they are genuine
              high-value pitches, not data errors.
            - The target is **imbalanced** (about 70% No / 30% Yes), so deal *rate* is used
              rather than raw counts when comparing groups.
            """
        )
    with c2:
        st.subheader("🧭 How to Use This Dashboard")
        st.markdown(
            """
            - **Univariate** – distribution of every individual variable.
            - **Bivariate** – deal outcome vs. one other variable, including **Profit vs Loss**,
              **City-wise** and **Industry-wise** views.
            - **Multivariate** – correlation, heatmaps and grouped comparisons combining 3+ variables.
            - Use the **sidebar filters** (season, industry, city, stage, profit/loss, channel)
              – every chart and KPI updates instantly.
            - Amounts are in **₹ Lakh** (1 Crore = 100 Lakh).
            """
        )

    st.subheader("🔍 Data Preview")
    st.dataframe(f.drop(columns=["deal_flag"]).head(100), hide_index=True)

# ----------------------------------------------------------------------------
# UNIVARIATE
# ----------------------------------------------------------------------------
elif page == "📈 Univariate":
    st.title("📈 Univariate Analysis")
    st.caption("Distribution of one variable at a time")
    kpi_row(f)

    tab1, tab2, tab3 = st.tabs(["Categorical", "Numeric", "Profit vs Loss & Deal Split"])

    with tab1:
        col = st.selectbox("Choose a categorical variable", CAT_COLS, format_func=lab)
        counts = f[col].value_counts().reset_index()
        counts.columns = [col, "pitches"]
        if col in ("season", "founder_count"):
            counts = counts.sort_values(col)
        elif col == "business_stage":
            counts[col] = pd.Categorical(counts[col], STAGE_ORDER, ordered=True)
            counts = counts.sort_values(col)
        counts["share_pct"] = (counts["pitches"] / counts["pitches"].sum() * 100).round(1)
        counts[col] = counts[col].astype(str)

        c1, c2 = st.columns([3, 1])
        with c1:
            horizontal = col in ("industry", "city")
            if horizontal:
                counts = counts.sort_values("pitches")
            fig = px.bar(counts, x="pitches" if horizontal else col,
                         y=col if horizontal else "pitches",
                         orientation="h" if horizontal else "v",
                         text="share_pct", color_discrete_sequence=[PRIMARY],
                         title=f"Number of pitches by {lab(col)}",
                         labels={"pitches": "Pitches", col: lab(col)})
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            st.plotly_chart(style(fig, 460))
        with c2:
            top = counts.sort_values("pitches", ascending=False).iloc[0]
            st.metric("Most common", str(top[col]))
            st.metric("Its share", f"{top['share_pct']}%")
            st.metric("Categories", counts[col].nunique())

    with tab2:
        col = st.selectbox("Choose a numeric variable", NUM_COLS, format_func=lab)
        s_all = f[col].dropna()
        if col == "deal_amount_lakh_inr":  # same as notebook: deals only (> 0)
            s_all = s_all[s_all > 0]
            st.caption("Deal amount is shown for funded startups only (amount > 0).")
        bins = st.slider("Number of bins", 10, 100, 40)
        s_view, view_note = clip_for_view(s_all)

        c1, c2 = st.columns(2)
        with c1:
            fig = histogram_fig(s_view, bins, f"Distribution of {lab(col)}", lab(col))
            st.plotly_chart(style(fig, 400))
        with c2:
            fig = px.box(s_view, x=s_view.name, color_discrete_sequence=[PRIMARY],
                         title=f"Boxplot of {lab(col)}", labels={col: lab(col)})
            st.plotly_chart(style(fig, 400))
        if view_note:
            st.caption(view_note)

        stats = s_all.describe(percentiles=[.25, .5, .75]).round(2).to_frame("value").T
        stats["skew"] = round(s_all.skew(), 2)
        st.markdown("**Summary statistics**")
        st.dataframe(stats, hide_index=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            pc = f["profit_status"].value_counts().reset_index()
            pc.columns = ["profit_status", "pitches"]
            fig = px.pie(pc, names="profit_status", values="pitches", hole=0.5,
                         color="profit_status", color_discrete_map=PROFIT_COLORS,
                         title="Profit vs Loss making startups")
            st.plotly_chart(style(fig, 380))
        with c2:
            dc = f["deal_made"].value_counts().reset_index()
            dc.columns = ["deal_made", "pitches"]
            fig = px.pie(dc, names="deal_made", values="pitches", hole=0.5,
                         color="deal_made", color_discrete_map=DEAL_COLORS,
                         title="Deal outcome (class balance)")
            st.plotly_chart(style(fig, 380))
        st.caption("Only about three in ten pitches end in a deal, so the target is imbalanced.")

# ----------------------------------------------------------------------------
# BIVARIATE
# ----------------------------------------------------------------------------
elif page == "🔗 Bivariate":
    st.title("🔗 Bivariate Analysis")
    st.caption("Relationship between deal outcome and one other variable")
    kpi_row(f)

    tabs = st.tabs(["Deal Rate Explorer", "Profit vs Loss", "City-wise",
                    "Industry-wise", "Numeric Relationships"])

    # --- Deal rate explorer -------------------------------------------------
    with tabs[0]:
        col = st.selectbox("Compare deal rate by", [c for c in CAT_COLS if c != "profit_status"],
                           format_func=lab)
        g = deal_rate_table(f, col)
        if col == "business_stage":
            g[col] = pd.Categorical(g[col], STAGE_ORDER, ordered=True)
            g = g.sort_values(col)
        elif col in ("season", "founder_count"):
            g = g.sort_values(col)
        else:
            g = g.sort_values("deal_rate", ascending=False)
        g[col] = g[col].astype(str)
        overall = f["deal_flag"].mean() * 100

        fig = px.bar(g, x=col, y="deal_rate", text="deal_rate",
                     color_discrete_sequence=[PRIMARY],
                     title=f"Deal rate (%) by {lab(col)}",
                     labels={"deal_rate": "Deal Rate (%)", col: lab(col)},
                     hover_data=["pitches", "deals"])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.add_hline(y=overall, line_dash="dash", line_color="grey",
                      annotation_text=f"Overall {overall:.1f}%")
        fig.update_yaxes(range=[0, max(g["deal_rate"].max() * 1.25, overall * 1.25)])
        st.plotly_chart(style(fig, 440))

        spread = g["deal_rate"].max() - g["deal_rate"].min()
        st.info(f"Spread between best and worst group: **{spread:.1f} percentage points**. "
                "Small spreads mean the variable is not a strong driver on its own.")
        with st.expander("View table"):
            st.dataframe(g, hide_index=True)

    # --- Profit vs Loss -----------------------------------------------------
    with tabs[1]:
        g = deal_rate_table(f, "profit_status")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(g, x="profit_status", y="deal_rate", text="deal_rate",
                         color="profit_status", color_discrete_map=PROFIT_COLORS,
                         title="Deal rate (%) – Profit vs Loss",
                         labels={"deal_rate": "Deal Rate (%)", "profit_status": ""})
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(showlegend=False)
            fig.update_yaxes(range=[0, g["deal_rate"].max() * 1.25])
            st.plotly_chart(style(fig, 400))
        with c2:
            ct = f.groupby(["profit_status", "deal_made"]).size().reset_index(name="pitches")
            fig = px.bar(ct, x="profit_status", y="pitches", color="deal_made", barmode="group",
                         color_discrete_map=DEAL_COLORS,
                         title="Pitches by profit status and deal outcome",
                         labels={"profit_status": "", "pitches": "Pitches"})
            st.plotly_chart(style(fig, 400))

        st.markdown("**Profit vs Loss across business stage**")
        pl = f.groupby(["business_stage", "profit_status"], observed=True).agg(
            deal_rate=("deal_flag", "mean")).reset_index()
        pl["deal_rate"] = (pl["deal_rate"] * 100).round(2)
        pl["business_stage"] = pd.Categorical(pl["business_stage"], STAGE_ORDER, ordered=True)
        pl = pl.sort_values("business_stage")
        fig = px.bar(pl, x="business_stage", y="deal_rate", color="profit_status", barmode="group",
                     color_discrete_map=PROFIT_COLORS, text="deal_rate",
                     labels={"deal_rate": "Deal Rate (%)", "business_stage": "Business Stage"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(style(fig, 400))

        p = g.set_index("profit_status")["deal_rate"]
        if {"Profit", "Loss"} <= set(p.index):
            st.success(f"Profit-making startups close deals **{p['Profit']:.1f}%** of the time versus "
                       f"**{p['Loss']:.1f}%** for loss-making ones – a gap of "
                       f"**{p['Profit'] - p['Loss']:.1f} points** within the current selection.")

    # --- City-wise ----------------------------------------------------------
    with tabs[2]:
        g = deal_rate_table(f, "city")
        metric = st.radio("Rank cities by", ["Deal Rate (%)", "Number of Pitches"], horizontal=True)
        n_city = int(g["city"].nunique())
        top_n = st.slider("Number of cities", 1, n_city, min(10, n_city)) if n_city > 1 else 1
        key = "deal_rate" if metric.startswith("Deal") else "pitches"
        gg = g.sort_values(key, ascending=False).head(top_n).sort_values(key)
        fig = px.bar(gg, x=key, y="city", orientation="h", text=key,
                     color=key, color_continuous_scale="Blues",
                     title=f"Top {top_n} cities by {metric}",
                     labels={"deal_rate": "Deal Rate (%)", "pitches": "Pitches", "city": ""},
                     hover_data=["pitches", "deals", "deal_rate"])
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style(fig, 460))

        st.markdown("**Profit vs Loss mix by city**")
        pm = f.groupby(["city", "profit_status"]).size().reset_index(name="pitches")
        order = g.sort_values("pitches", ascending=False)["city"].head(top_n).tolist()
        pm = pm[pm["city"].isin(order)]
        fig = px.bar(pm, x="city", y="pitches", color="profit_status", barmode="stack",
                     category_orders={"city": order}, color_discrete_map=PROFIT_COLORS,
                     labels={"city": "", "pitches": "Pitches"})
        st.plotly_chart(style(fig, 400))

    # --- Industry-wise ------------------------------------------------------
    with tabs[3]:
        g = deal_rate_table(f, "industry").sort_values("deal_rate")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(g, x="deal_rate", y="industry", orientation="h", text="deal_rate",
                         color_discrete_sequence=[PRIMARY],
                         title="Deal rate (%) by industry",
                         labels={"deal_rate": "Deal Rate (%)", "industry": ""},
                         hover_data=["pitches", "deals"])
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_xaxes(range=[0, g["deal_rate"].max() * 1.2])
            st.plotly_chart(style(fig, 480))
        with c2:
            fd = f[f["deal_flag"] == 1]
            fa = fd.groupby("industry")["deal_amount_lakh_inr"].median().reset_index() \
                   .sort_values("deal_amount_lakh_inr")
            fig = px.bar(fa, x="deal_amount_lakh_inr", y="industry", orientation="h",
                         text=fa["deal_amount_lakh_inr"].round(1),
                         color_discrete_sequence=["#2E8B57"],
                         title="Median deal amount (₹ Lakh), funded startups",
                         labels={"deal_amount_lakh_inr": "Median Deal (₹ Lakh)", "industry": ""})
            fig.update_traces(textposition="outside")
            st.plotly_chart(style(fig, 480))

        st.markdown("**Profit share by industry**")
        ps = f.groupby("industry")["profit_status"].apply(lambda s: (s == "Profit").mean() * 100) \
              .round(1).reset_index(name="profit_pct").sort_values("profit_pct")
        fig = px.bar(ps, x="profit_pct", y="industry", orientation="h", text="profit_pct",
                     color_discrete_sequence=["#2E8B57"],
                     labels={"profit_pct": "% of startups in profit", "industry": ""})
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_xaxes(range=[0, 100])
        st.plotly_chart(style(fig, 440))

    # --- Numeric relationships ---------------------------------------------
    with tabs[4]:
        st.markdown("**Numeric variable by deal outcome**")
        num = st.selectbox("Numeric variable",
                           ["asking_amount_lakh_inr", "equity_offered_pct",
                            "annual_sales_lakh_inr", "profit_margin_pct"], format_func=lab)
        log_y = st.checkbox("Log scale", value=num != "profit_margin_pct" and num != "equity_offered_pct")
        fig = px.box(f.dropna(subset=[num]), x="deal_made", y=num, color="deal_made",
                     log_y=log_y, color_discrete_map=DEAL_COLORS,
                     title=f"{lab(num)} by deal outcome",
                     labels={"deal_made": "Deal Made", num: lab(num)})
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 420))
        med = f.groupby("deal_made")[num].median().round(2)
        st.caption("Median – " + " | ".join(f"{k}: {v:,}" for k, v in med.items()))

        st.markdown("**Scatter: two numeric variables**")
        c1, c2 = st.columns(2)
        xv = c1.selectbox("X axis", NUM_COLS[:4], index=0, format_func=lab, key="bx")
        yv = c2.selectbox("Y axis", NUM_COLS[:4], index=2, format_func=lab, key="by")
        sd = f.dropna(subset=[xv, yv])
        sd = sd.sample(min(4000, len(sd)), random_state=42)
        fig = px.scatter(sd, x=xv, y=yv, opacity=0.45, color_discrete_sequence=[PRIMARY],
                         log_x=xv in ("annual_sales_lakh_inr", "asking_amount_lakh_inr"),
                         log_y=yv in ("annual_sales_lakh_inr", "asking_amount_lakh_inr"),
                         labels={xv: lab(xv), yv: lab(yv)},
                         title=f"{lab(yv)} vs {lab(xv)} (random sample of {len(sd):,})")
        st.plotly_chart(style(fig, 440))

# ----------------------------------------------------------------------------
# MULTIVARIATE
# ----------------------------------------------------------------------------
else:
    st.title("🧩 Multivariate Analysis")
    st.caption("Patterns involving three or more variables at once")
    kpi_row(f)

    tabs = st.tabs(["Correlation", "Heatmap", "Grouped Comparison", "Scatter by Outcome"])

    with tabs[0]:
        method = st.radio("Method", ["pearson", "spearman"], horizontal=True,
                          help="Spearman is rank-based and less affected by outliers.")
        corr = f[NUM_COLS + ["deal_flag"]].rename(columns=LABELS | {"deal_flag": "Deal (1/0)"}) \
                 .corr(method=method).round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto", zmin=-1, zmax=1,
                        color_continuous_scale="RdBu_r",
                        title=f"{method.capitalize()} correlation of numeric features")
        st.plotly_chart(style(fig, 520))
        st.caption("Correlations among financial metrics are mostly weak, so each metric carries "
                   "largely independent information about a pitch.")

    with tabs[1]:
        c1, c2 = st.columns(2)
        row = c1.selectbox("Rows", ["industry", "city", "business_stage", "season"], index=0, format_func=lab)
        colv = c2.selectbox("Columns", ["business_stage", "profit_status", "sales_channel", "season"],
                            index=0, format_func=lab)
        if row == colv:
            st.warning("Choose different variables for rows and columns.")
        else:
            val = st.radio("Value", ["Deal rate (%)", "Number of pitches"], horizontal=True)
            if val.startswith("Deal"):
                pv = f.pivot_table(index=row, columns=colv, values="deal_flag", aggfunc="mean") * 100
            else:
                pv = f.pivot_table(index=row, columns=colv, values="deal_flag", aggfunc="size")
            if colv == "business_stage":
                pv = pv[[c for c in STAGE_ORDER if c in pv.columns]]
            if row == "business_stage":
                pv = pv.reindex([s for s in STAGE_ORDER if s in pv.index])
            fig = px.imshow(pv.round(1), text_auto=True, aspect="auto",
                            color_continuous_scale="YlGnBu",
                            title=f"{val}: {lab(row)} × {lab(colv)}",
                            labels={"color": val})
            st.plotly_chart(style(fig, 520))

    with tabs[2]:
        st.markdown("**Deal rate by business stage, profit status and sales channel**")
        g = f.groupby(["business_stage", "profit_status", "sales_channel"], observed=True) \
             .agg(deal_rate=("deal_flag", "mean"), pitches=("deal_flag", "size")).reset_index()
        g["deal_rate"] = (g["deal_rate"] * 100).round(2)
        g["business_stage"] = pd.Categorical(g["business_stage"], STAGE_ORDER, ordered=True)
        g = g.sort_values("business_stage")
        fig = px.bar(g, x="business_stage", y="deal_rate", color="profit_status", barmode="group",
                     facet_col="sales_channel", color_discrete_map=PROFIT_COLORS,
                     hover_data=["pitches"],
                     labels={"deal_rate": "Deal Rate (%)", "business_stage": "Stage",
                             "sales_channel": "Channel"})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(style(fig, 460))

        st.markdown("**Asking amount by industry, split by deal outcome**")
        sub = f.dropna(subset=["asking_amount_lakh_inr"])
        fig = px.box(sub, x="industry", y="asking_amount_lakh_inr", color="deal_made", log_y=True,
                     color_discrete_map=DEAL_COLORS,
                     labels={"asking_amount_lakh_inr": "Asking Amount (₹ Lakh, log)",
                             "industry": "", "deal_made": "Deal"})
        st.plotly_chart(style(fig, 460))

    with tabs[3]:
        sd = f.dropna(subset=["annual_sales_lakh_inr", "asking_amount_lakh_inr"])
        sd = sd.sample(min(5000, len(sd)), random_state=7)
        fig = px.scatter(sd, x="annual_sales_lakh_inr", y="asking_amount_lakh_inr",
                         color="deal_made", log_x=True, log_y=True, opacity=0.5,
                         color_discrete_map=DEAL_COLORS,
                         hover_data=["industry", "city", "business_stage"],
                         title=f"Annual sales vs asking amount by deal outcome (sample of {len(sd):,})",
                         labels={"annual_sales_lakh_inr": "Annual Sales (₹ Lakh, log)",
                                 "asking_amount_lakh_inr": "Asking Amount (₹ Lakh, log)",
                                 "deal_made": "Deal"})
        st.plotly_chart(style(fig, 520))
        st.caption("Deals and non-deals are mixed throughout the space – sales and ask alone do not "
                   "decide the outcome.")

st.markdown("---")
st.caption("Startup Funding & Deal Outcome Analysis · Built with Streamlit & Plotly")
