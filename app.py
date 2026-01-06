import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Executive Sales Performance & Customer Insights Dashboard",
    page_icon="📊",
    layout="wide",
)

CURRENCY = "$"


def fmt_currency(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{CURRENCY}{x:,.0f}"


def fmt_currency_2(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{CURRENCY}{x:,.2f}"


def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.1%}"


def comparable_previous_period(start: pd.Timestamp, end: pd.Timestamp):
    """Previous comparable period = same length immediately preceding the selected range."""
    days = (end - start).days + 1
    prev_end = start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=days - 1)
    return prev_start, prev_end


@st.cache_data(show_spinner=False)
def load_data():
    orders = pd.read_csv("orders.csv", parse_dates=["order_date"])
    customers = pd.read_csv("customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv("products.csv")
    sales = pd.read_csv(
        "sales.csv",
        parse_dates=["order_date"],
        dtype={"is_return": "int64"},
    )
    # Normalize types
    sales["discount_pct"] = sales["discount_pct"].astype(float)
    sales["quantity"] = sales["quantity"].astype(int)
    sales["is_return"] = sales["is_return"].astype(int)
    return orders, customers, products, sales


def data_quality_checks(orders, customers, products, sales):
    issues = []

    # Orphan checks
    missing_orders = set(sales["order_id"].unique()) - set(orders["order_id"].unique())
    missing_customers = set(sales["customer_id"].unique()) - set(customers["customer_id"].unique())
    missing_products = set(sales["product_id"].unique()) - set(products["product_id"].unique())

    if missing_orders:
        issues.append(f"Orphan keys: {len(missing_orders)} sales rows reference missing order_id(s).")
    if missing_customers:
        issues.append(f"Orphan keys: {len(missing_customers)} sales rows reference missing customer_id(s).")
    if missing_products:
        issues.append(f"Orphan keys: {len(missing_products)} sales rows reference missing product_id(s).")

    # Reconciliation checks
    # Revenue formula: unit_price * quantity * (1 - discount_pct), with returns represented by negative line_revenue
    calc_rev = sales["unit_price"] * sales["quantity"] * (1 - sales["discount_pct"])
    calc_rev = np.where(sales["is_return"].eq(1), -np.abs(calc_rev), calc_rev)

    rev_diff = (sales["line_revenue"] - calc_rev).abs().max()
    if rev_diff > 0.02:
        issues.append(f"Revenue reconciliation: max absolute deviation {rev_diff:.4f} (expected <= 0.02).")

    calc_profit = sales["line_revenue"] - (sales["unit_cost"] * sales["quantity"]) - sales["shipping_cost_allocated"]
    calc_profit = np.where(sales["is_return"].eq(1), -np.abs(calc_profit), calc_profit)

    prof_diff = (sales["line_profit"] - calc_profit).abs().max()
    if prof_diff > 0.02:
        issues.append(f"Profit reconciliation: max absolute deviation {prof_diff:.4f} (expected <= 0.02).")

    # Missingness summary
    miss = (sales.isna().mean().sort_values(ascending=False))
    high_miss = miss[miss > 0].head(10)
    if len(high_miss) > 0:
        issues.append("Missing values detected in sales (top columns): " + ", ".join([f"{k}={v:.1%}" for k, v in high_miss.items()]))

    return issues


def compute_customer_first_purchase(sales: pd.DataFrame) -> pd.DataFrame:
    """Derive first purchase date from NON-return lines to avoid defining cohorts on returns."""
    base = sales[sales["is_return"].eq(0)].copy()
    first = base.groupby("customer_id", as_index=False)["order_date"].min().rename(columns={"order_date": "first_purchase_date"})
    return first


def add_customer_spend_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Segment customers by spend within the filtered dataset (dynamic, executive-useful)."""
    cust_rev = df.groupby("customer_id", as_index=False)["line_revenue"].sum()
    # Use only positive revenue for spend segmentation
    cust_rev["pos_revenue"] = cust_rev["line_revenue"].clip(lower=0)
    q1, q2, q3 = cust_rev["pos_revenue"].quantile([0.25, 0.5, 0.75]).values

    def seg(x):
        if x <= q1:
            return "Low"
        if x <= q3:
            return "Mid"
        return "High"

    cust_rev["spend_segment"] = cust_rev["pos_revenue"].apply(seg)
    return df.merge(cust_rev[["customer_id", "spend_segment"]], on="customer_id", how="left")


def kpi_block(current: dict, previous: dict):
    cols = st.columns(7)

    def delta(cur, prev):
        if prev is None or prev == 0 or (isinstance(prev, float) and np.isnan(prev)):
            return None
        return cur - prev

    def delta_pct(cur, prev):
        if prev is None or prev == 0 or (isinstance(prev, float) and np.isnan(prev)):
            return None
        return (cur / prev) - 1

    cols[0].metric("Total Revenue", fmt_currency(current["revenue"]), None if previous["revenue"] == 0 else fmt_currency(delta(current["revenue"], previous["revenue"])))
    cols[1].metric("Total Profit", fmt_currency(current["profit"]), None if previous["profit"] == 0 else fmt_currency(delta(current["profit"], previous["profit"])))
    cols[2].metric("AOV", fmt_currency_2(current["aov"]), None if previous["aov"] == 0 else fmt_currency_2(delta(current["aov"], previous["aov"])))
    cols[3].metric("Total Orders", f'{int(current["orders"]):,}', None if previous["orders"] == 0 else f'{int(delta(current["orders"], previous["orders"])):,}')
    cols[4].metric("Profit Margin (%)", fmt_pct(current["margin"]), None if previous["margin"] == 0 else fmt_pct(delta(current["margin"], previous["margin"])))
    cols[5].metric("Revenue Growth %", fmt_pct(current["rev_growth"]), None if previous["rev_growth"] == 0 else fmt_pct(delta(current["rev_growth"], previous["rev_growth"])))
    cols[6].metric("Return Rate %", fmt_pct(current["return_rate"]), None if previous["return_rate"] == 0 else fmt_pct(delta(current["return_rate"], previous["return_rate"])))


def compute_kpis(df: pd.DataFrame):
    # Use net revenue/profit (returns are negative line_revenue/line_profit).
    revenue = df["line_revenue"].sum()
    profit = df["line_profit"].sum()
    orders = df.loc[df["is_return"].eq(0), "order_id"].nunique()
    aov = revenue / orders if orders > 0 else 0.0
    margin = (profit / revenue) if revenue != 0 else 0.0  # weighted margin

    # Return rate definition (README): return lines / total lines (within selection).
    total_lines = len(df)
    return_lines = int(df["is_return"].sum())
    return_rate = return_lines / total_lines if total_lines > 0 else 0.0

    return dict(
        revenue=float(revenue),
        profit=float(profit),
        orders=int(orders),
        aov=float(aov),
        margin=float(margin),
        return_rate=float(return_rate),
    )


def insights_panel(df: pd.DataFrame):
    st.subheader("Key Insights (Auto-Computed)")
    insights = []

    if df.empty:
        st.info("No data for current filters.")
        return

    # Region best/worst by revenue and margin
    reg = df.groupby("region", as_index=False).agg(revenue=("line_revenue", "sum"), profit=("line_profit", "sum"))
    reg["margin"] = np.where(reg["revenue"] != 0, reg["profit"] / reg["revenue"], np.nan)
    best_rev = reg.sort_values("revenue", ascending=False).head(1)
    worst_margin = reg.sort_values("margin", ascending=True).head(1)

    if len(best_rev):
        insights.append(f"Top region by revenue: **{best_rev.iloc[0]['region']}** ({fmt_currency(best_rev.iloc[0]['revenue'])}).")
    if len(worst_margin):
        insights.append(f"Lowest margin region: **{worst_margin.iloc[0]['region']}** ({fmt_pct(worst_margin.iloc[0]['margin'])}).")

    # Highest return categories/products
    ret = df.groupby("category", as_index=False).agg(lines=("sales_line_id", "count"), returns=("is_return", "sum"))
    ret["return_rate"] = np.where(ret["lines"] > 0, ret["returns"] / ret["lines"], 0)
    top_ret_cat = ret.sort_values("return_rate", ascending=False).head(1)
    if len(top_ret_cat):
        insights.append(f"Highest return-rate category: **{top_ret_cat.iloc[0]['category']}** ({fmt_pct(top_ret_cat.iloc[0]['return_rate'])}).")

    prod_ret = df.groupby("product_id", as_index=False).agg(lines=("sales_line_id", "count"), returns=("is_return", "sum"))
    prod_ret["return_rate"] = np.where(prod_ret["lines"] > 0, prod_ret["returns"] / prod_ret["lines"], 0)
    top_ret_prod = prod_ret.sort_values("return_rate", ascending=False).head(1)
    if len(top_ret_prod):
        insights.append(f"Highest return-rate product (by lines): **{top_ret_prod.iloc[0]['product_id']}** ({fmt_pct(top_ret_prod.iloc[0]['return_rate'])}).")

    # Customer concentration: top 10% customers revenue share
    cust = df.groupby("customer_id", as_index=False)["line_revenue"].sum()
    cust["pos_revenue"] = cust["line_revenue"].clip(lower=0)
    if len(cust) >= 10:
        cutoff = max(1, int(np.ceil(0.10 * len(cust))))
        share = cust.sort_values("pos_revenue", ascending=False).head(cutoff)["pos_revenue"].sum() / max(cust["pos_revenue"].sum(), 1e-9)
        insights.append(f"Customer concentration: top 10% customers contribute **{fmt_pct(share)}** of positive revenue.")

    # Discount vs margin correlation (line-level)
    ddf = df[df["is_return"].eq(0)].copy()
    ddf["line_margin"] = np.where(ddf["line_revenue"] != 0, ddf["line_profit"] / ddf["line_revenue"], np.nan)
    corr = ddf[["discount_pct", "line_margin"]].dropna().corr().iloc[0, 1] if len(ddf) > 2 else np.nan
    if not np.isnan(corr):
        direction = "negative" if corr < 0 else "positive"
        insights.append(f"Discount sensitivity: correlation(discount%, margin%) is **{corr:.2f}** ({direction}).")

    # High-revenue / low-margin attention items (product)
    prod = df.groupby(["product_id"], as_index=False).agg(revenue=("line_revenue", "sum"), profit=("line_profit", "sum"), orders=("order_id", "nunique"))
    prod["margin"] = np.where(prod["revenue"] != 0, prod["profit"] / prod["revenue"], np.nan)
    rev_thresh = prod["revenue"].quantile(0.75)
    low_margin_thresh = prod["margin"].quantile(0.25)
    attention = prod[(prod["revenue"] >= rev_thresh) & (prod["margin"] <= low_margin_thresh)].sort_values("revenue", ascending=False).head(3)
    if len(attention) > 0:
        items = ", ".join([f"{r.product_id} ({fmt_currency(r.revenue)}, {fmt_pct(r.margin)})" for r in attention.itertuples(index=False)])
        insights.append(f"High-revenue / low-margin products needing attention: {items}.")

    # Render
    if len(insights) < 5:
        st.warning("Limited insight generation due to small filtered dataset; broaden filters for richer insights.")
    for i in insights[:10]:
        st.markdown(f"- {i}")


# ---------------- UI ----------------
st.title("Executive Sales Performance & Customer Insights Dashboard")
st.caption("Decision-grade view of revenue, profit, margin, retention, discount discipline, returns risk, and portfolio mix.")

orders, customers, products, sales = load_data()

issues = data_quality_checks(orders, customers, products, sales)
if issues:
    with st.expander("Data Quality Warnings", expanded=True):
        for msg in issues:
            st.warning(msg)
else:
    st.success("Data validation passed: referential integrity + revenue/profit reconciliation checks OK.", icon="✅")

# Enrich
cust_first = compute_customer_first_purchase(sales)
sales = sales.merge(customers[["customer_id", "segment"]], on="customer_id", how="left")
sales = sales.merge(products[["product_id", "product_name", "sub_category"]], on="product_id", how="left")
sales = sales.merge(cust_first, on="customer_id", how="left")

# Sidebar filters
st.sidebar.header("Filters")
min_date = sales["order_date"].min().date()
max_date = sales["order_date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)

regions = ["All"] + sorted(sales["region"].dropna().unique().tolist())
categories = ["All"] + sorted(sales["category"].dropna().unique().tolist())
segments = ["All"] + sorted(sales["segment"].dropna().unique().tolist())
channels = ["All"] + sorted(sales["channel"].dropna().unique().tolist())

region_sel = st.sidebar.selectbox("Region", regions, index=0)
cat_sel = st.sidebar.selectbox("Product category", categories, index=0)
seg_sel = st.sidebar.selectbox("Customer segment", segments, index=0)
channel_sel = st.sidebar.selectbox("Channel", channels, index=0)

include_returns = st.sidebar.toggle("Include returns (net view)", value=True, help="ON = include return lines (negative revenue/profit). OFF = sales only.")
show_only_returns = st.sidebar.toggle("Show only returns", value=False)

start = pd.to_datetime(date_range[0])
end = pd.to_datetime(date_range[1])

df = sales[(sales["order_date"] >= start) & (sales["order_date"] <= end)].copy()
if region_sel != "All":
    df = df[df["region"].eq(region_sel)]
if cat_sel != "All":
    df = df[df["category"].eq(cat_sel)]
if seg_sel != "All":
    df = df[df["segment"].eq(seg_sel)]
if channel_sel != "All":
    df = df[df["channel"].eq(channel_sel)]

if show_only_returns:
    df = df[df["is_return"].eq(1)]
elif not include_returns:
    df = df[df["is_return"].eq(0)]

# Dynamic spend segment (for histogram)
df = add_customer_spend_segment(df)

prev_start, prev_end = comparable_previous_period(start, end)
df_prev = sales[(sales["order_date"] >= prev_start) & (sales["order_date"] <= prev_end)].copy()
if region_sel != "All":
    df_prev = df_prev[df_prev["region"].eq(region_sel)]
if cat_sel != "All":
    df_prev = df_prev[df_prev["category"].eq(cat_sel)]
if seg_sel != "All":
    df_prev = df_prev[df_prev["segment"].eq(seg_sel)]
if channel_sel != "All":
    df_prev = df_prev[df_prev["channel"].eq(channel_sel)]
if show_only_returns:
    df_prev = df_prev[df_prev["is_return"].eq(1)]
elif not include_returns:
    df_prev = df_prev[df_prev["is_return"].eq(0)]

# KPIs
kpi_cur = compute_kpis(df)
kpi_prev = compute_kpis(df_prev)

# Revenue growth % vs previous comparable period
kpi_cur["rev_growth"] = (kpi_cur["revenue"] / kpi_prev["revenue"] - 1) if kpi_prev["revenue"] != 0 else np.nan
kpi_prev["rev_growth"] = np.nan  # not meaningful for prior period delta display

st.markdown("### Executive KPIs")
kpi_block(kpi_cur, kpi_prev)

st.divider()

# ---------------- Required visuals ----------------
st.markdown("## Sales & Profitability Diagnostics")

# 1) Monthly Sales Trend – Line chart
m = df.copy()
m["month"] = m["order_date"].dt.to_period("M").dt.to_timestamp()
m_month = m.groupby("month", as_index=False).agg(revenue=("line_revenue", "sum"), profit=("line_profit", "sum"))
fig1 = px.line(
    m_month.sort_values("month"),
    x="month",
    y="revenue",
    markers=True,
    title="Monthly Sales Trend (Net Revenue)",
    labels={"month": "Month", "revenue": "Revenue"},
)
fig1.update_yaxes(tickprefix=CURRENCY, tickformat=",.0f")
fig1.update_layout(height=380, margin=dict(l=10, r=10, t=55, b=10))
st.plotly_chart(fig1, use_container_width=True)
st.markdown(
    "- Tracks seasonality and campaign impact over time.\n"
    "- Use spikes to validate promotion effectiveness; use dips to investigate demand, supply, or pricing issues.\n"
    "- Compare with profit trend (below) to ensure growth is profitable, not discount-driven."
)

# 2) Top Products by Revenue – Bar chart
top_prod = df.groupby(["product_id", "product_name"], as_index=False)["line_revenue"].sum().sort_values("line_revenue", ascending=False).head(10)
fig2 = px.bar(
    top_prod,
    x="line_revenue",
    y="product_name",
    orientation="h",
    title="Top Products by Revenue (Top 10)",
    labels={"line_revenue": "Revenue", "product_name": "Product"},
)
fig2.update_xaxes(tickprefix=CURRENCY, tickformat=",.0f")
fig2.update_layout(height=420, yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=55, b=10))
st.plotly_chart(fig2, use_container_width=True)
st.markdown(
    "- Identifies portfolio revenue concentration and “hero” SKUs.\n"
    "- High revenue does not imply high profit—validate with margin charts and quadrant analysis.\n"
    "- Use for inventory and assortment prioritization."
)

# 3) Region-wise Sales Performance – Bar chart
reg = df.groupby("region", as_index=False).agg(revenue=("line_revenue", "sum"), profit=("line_profit", "sum"))
reg["margin"] = np.where(reg["revenue"] != 0, reg["profit"] / reg["revenue"], np.nan)
fig3 = px.bar(
    reg.sort_values("revenue", ascending=False),
    x="region",
    y="revenue",
    title="Region-wise Sales Performance (Revenue)",
    labels={"region": "Region", "revenue": "Revenue"},
    text_auto=True,
)
fig3.update_yaxes(tickprefix=CURRENCY, tickformat=",.0f")
fig3.update_layout(height=380, margin=dict(l=10, r=10, t=55, b=10))
st.plotly_chart(fig3, use_container_width=True)
st.markdown(
    "- Highlights which geographies drive topline performance.\n"
    "- Pair with margin (see insights + category margin) to distinguish scale vs profitability.\n"
    "- Supports resource allocation (sales capacity, marketing, fulfillment)."
)

# 4) Customer Segmentation by Spend – Histogram
cust_spend = df.groupby("customer_id", as_index=False)["line_revenue"].sum()
cust_spend["pos_revenue"] = cust_spend["line_revenue"].clip(lower=0)
fig4 = px.histogram(
    cust_spend,
    x="pos_revenue",
    nbins=20,
    title="Customer Segmentation by Spend (Histogram of Positive Revenue per Customer)",
    labels={"pos_revenue": "Customer Spend (Revenue)"},
)
fig4.update_xaxes(tickprefix=CURRENCY, tickformat=",.0f")
fig4.update_layout(height=380, margin=dict(l=10, r=10, t=55, b=10))
st.plotly_chart(fig4, use_container_width=True)
st.markdown(
    "- Shows whether revenue is broad-based or concentrated among a few customers.\n"
    "- A heavy tail suggests key-account risk and the need for retention/upsell strategies.\n"
    "- Use alongside the “top 10% share” insight to quantify concentration."
)

# 5) Profit Margin by Product Category – Bar chart
cat = df.groupby("category", as_index=False).agg(revenue=("line_revenue", "sum"), profit=("line_profit", "sum"))
cat["profit_margin"] = np.where(cat["revenue"] != 0, cat["profit"] / cat["revenue"], np.nan)
fig5 = px.bar(
    cat.sort_values("profit_margin", ascending=False),
    x="category",
    y="profit_margin",
    title="Profit Margin by Product Category (Weighted Margin = Profit / Revenue)",
    labels={"category": "Category", "profit_margin": "Profit Margin"},
    text=cat["profit_margin"].map(lambda x: f"{x:.1%}" if pd.notnull(x) else ""),
)
fig5.update_yaxes(tickformat=".0%")
fig5.update_layout(height=380, margin=dict(l=10, r=10, t=55, b=10))
st.plotly_chart(fig5, use_container_width=True)
st.markdown(
    "- Identifies categories that create (or destroy) economic value.\n"
    "- If high-revenue categories have weak margins, prioritize pricing, cost-to-serve, or discount policy.\n"
    "- Use for category strategy and promotion planning."
)

st.divider()

# ---------------- Additional executive visuals (non-bar focus) ----------------
st.markdown("## Executive Diagnostic Views")

# A) Profit vs Revenue Quadrant (Scatter/Bubble) — REQUIRED
quad = df.groupby(["product_id", "product_name", "category"], as_index=False).agg(
    revenue=("line_revenue", "sum"),
    profit=("line_profit", "sum"),
    orders=("order_id", "nunique"),
)
quad["margin"] = np.where(quad["revenue"] != 0, quad["profit"] / quad["revenue"], np.nan)
figA = px.scatter(
    quad,
    x="revenue",
    y="margin",
    size="orders",
    color="category",
    hover_name="product_name",
    title="Profit vs Revenue Quadrant (Products) — Size = Orders, Color = Category",
    labels={"revenue": "Revenue", "margin": "Profit Margin"},
)
figA.update_xaxes(tickprefix=CURRENCY, tickformat=",.0f")
figA.update_yaxes(tickformat=".0%")
figA.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
st.plotly_chart(figA, use_container_width=True)
st.markdown(
    "- High revenue + low margin points indicate pricing/cost issues despite scale.\n"
    "- Negative/low-margin bubbles warrant immediate action: discount discipline, vendor renegotiation, or SKU rationalization.\n"
    "- Compare categories to detect structural margin gaps."
)

# B) Customer Retention / Repeat Behavior — REQUIRED (Repeat purchase rate trend)
tmp = df[df["is_return"].eq(0)].copy()
tmp["month"] = tmp["order_date"].dt.to_period("M").dt.to_timestamp()
cust_month = tmp.groupby(["month", "customer_id"], as_index=False)["order_id"].nunique()
month_total_customers = cust_month.groupby("month", as_index=False)["customer_id"].nunique().rename(columns={"customer_id": "customers"})
# Repeat customer = customer who purchased previously (first_purchase_date < month start)
tmp_first = tmp.groupby("customer_id", as_index=False)["order_date"].min().rename(columns={"order_date": "first_purchase_date"})
cust_month = cust_month.merge(tmp_first, on="customer_id", how="left")
cust_month["is_repeat_in_month"] = cust_month["first_purchase_date"] < cust_month["month"]
repeat_by_month = cust_month.groupby("month", as_index=False).agg(repeat_customers=("is_repeat_in_month", "sum"))
rep = month_total_customers.merge(repeat_by_month, on="month", how="left")
rep["repeat_rate"] = np.where(rep["customers"] > 0, rep["repeat_customers"] / rep["customers"], np.nan)

figB = px.line(
    rep.sort_values("month"),
    x="month",
    y="repeat_rate",
    markers=True,
    title="Repeat Purchase Rate Trend (Monthly)",
    labels={"month": "Month", "repeat_rate": "Repeat Rate"},
)
figB.update_yaxes(tickformat=".0%")
figB.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=10))
st.plotly_chart(figB, use_container_width=True)
st.markdown(
    "- Measures retention health: increasing repeat rate implies stronger customer stickiness and lower acquisition dependency.\n"
    "- If repeat rate falls during heavy promo months, discounting may be attracting low-quality, non-returning buyers.\n"
    "- Use to guide CRM, loyalty, and post-purchase initiatives."
)

# C) Discount vs Margin Relationship (Scatter + Trendline) — REQUIRED
d = df[df["is_return"].eq(0)].copy()
d = d[d["line_revenue"] > 0]
d["line_margin"] = np.where(d["line_revenue"] != 0, d["line_profit"] / d["line_revenue"], np.nan)
figC = px.scatter(
    d,
    x="discount_pct",
    y="line_margin",
    color="category",
    trendline="ols",
    title="Discount % vs Line Profit Margin (Scatter + Trendline)",
    labels={"discount_pct": "Discount %", "line_margin": "Line Profit Margin"},
    hover_data=["product_name", "region", "quantity", "unit_price"],
)
figC.update_xaxes(tickformat=".0%")
figC.update_yaxes(tickformat=".0%")
figC.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
st.plotly_chart(figC, use_container_width=True)
st.markdown(
    "- Quantifies whether discounting is eroding profitability and where the inflection point occurs.\n"
    "- A steep negative trend suggests tightening discount guardrails or introducing margin-based approval thresholds.\n"
    "- Use category-level patterns to tailor promo strategy (not one-size-fits-all)."
)

st.divider()

# ---------------- Drill-down table + export ----------------
st.markdown("## Drill-Down: Transactions (Filtered)")
table_cols = [
    "sales_line_id", "order_id", "order_date", "customer_id", "segment",
    "region", "channel", "product_id", "product_name", "category",
    "quantity", "unit_price", "unit_cost", "discount_pct", "shipping_cost_allocated",
    "is_return", "line_revenue", "line_profit"
]
view = df[table_cols].sort_values("order_date", ascending=False)

st.dataframe(
    view,
    use_container_width=True,
    hide_index=True,
    column_config={
        "discount_pct": st.column_config.NumberColumn(format="%.0f%%"),
        "unit_price": st.column_config.NumberColumn(format=f"{CURRENCY}%.2f"),
        "unit_cost": st.column_config.NumberColumn(format=f"{CURRENCY}%.2f"),
        "shipping_cost_allocated": st.column_config.NumberColumn(format=f"{CURRENCY}%.2f"),
        "line_revenue": st.column_config.NumberColumn(format=f"{CURRENCY}%.2f"),
        "line_profit": st.column_config.NumberColumn(format=f"{CURRENCY}%.2f"),
        "is_return": st.column_config.NumberColumn(format="%d"),
    },
)

buf = io.StringIO()
view.to_csv(buf, index=False)
st.download_button(
    "Download filtered transactions as CSV",
    data=buf.getvalue(),
    file_name="filtered_transactions.csv",
    mime="text/csv",
)

st.divider()
insights_panel(df)