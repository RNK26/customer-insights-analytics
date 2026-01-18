"""Streamlit dashboard over the retail warehouse.

Run:  streamlit run app.py
Needs data/retail.duckdb, which build_db.py produces.
"""

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = "data/retail.duckdb"

st.set_page_config(page_title="Online Retail II - Commercial BI", page_icon="📊",
                   layout="wide")


# Without the cache every widget change re-hits DuckDB and the app drags.
@st.cache_data
def run_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        return con.execute(sql, params).df()
    finally:
        con.close()


@st.cache_data
def get_filter_options() -> tuple[list[str], pd.Timestamp, pd.Timestamp]:
    df = run_query(
        "SELECT MIN(InvoiceDate) AS lo, MAX(InvoiceDate) AS hi FROM sales"
    )
    countries = run_query(
        "SELECT Country FROM sales GROUP BY Country ORDER BY SUM(Revenue) DESC"
    )["Country"].tolist()
    return countries, df["lo"].iloc[0], df["hi"].iloc[0]


countries, min_date, max_date = get_filter_options()

st.sidebar.header("Filters")

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)
# date_input hands back a single date until the second one is picked.
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date.date(), max_date.date()

selected = st.sidebar.multiselect(
    "Countries (blank = all)", options=countries, default=[]
)

st.sidebar.caption(
    "Data: UCI Online Retail II (CC BY 4.0). Cancelled orders, non-positive "
    "quantities/prices and duplicates already removed in build_db.py."
)

# One WHERE clause, reused by every query below. Values go in as bound
# parameters rather than formatted into the string.
where = "WHERE InvoiceDate BETWEEN ? AND ?"
params: list = [str(start_date), str(end_date) + " 23:59:59"]
if selected:
    placeholders = ", ".join("?" for _ in selected)
    where += f" AND Country IN ({placeholders})"
    params += selected
params = tuple(params)

st.title("Online Retail II - Commercial Performance")

kpis = run_query(
    f"""
    SELECT
        SUM(Revenue)                                AS revenue,
        COUNT(DISTINCT Invoice)                     AS orders,
        SUM(Revenue) / COUNT(DISTINCT Invoice)      AS avg_order_value,
        COUNT(DISTINCT CustomerID)                  AS customers
    FROM sales {where}
    """,
    params,
)

top_country = run_query(
    f"""
    SELECT Country, SUM(Revenue) AS revenue
    FROM sales {where}
    GROUP BY Country ORDER BY revenue DESC LIMIT 1
    """,
    params,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue", f"£{kpis['revenue'].iloc[0]:,.0f}")
c2.metric("Orders", f"{int(kpis['orders'].iloc[0]):,}")
c3.metric("Avg order value", f"£{kpis['avg_order_value'].iloc[0]:,.2f}")
c4.metric(
    "Top market",
    top_country["Country"].iloc[0] if len(top_country) else "-",
    f"£{top_country['revenue'].iloc[0]:,.0f}" if len(top_country) else "",
)

st.subheader("Monthly revenue and cumulative total")

trend = run_query(
    f"""
    WITH monthly AS (
        SELECT DATE_TRUNC('month', InvoiceDate) AS month,
               SUM(Revenue) AS monthly_revenue
        FROM sales {where}
        GROUP BY 1
    )
    SELECT month, monthly_revenue,
           SUM(monthly_revenue) OVER (
               ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
           ) AS running_total
    FROM monthly ORDER BY month
    """,
    params,
)

fig = px.bar(trend, x="month", y="monthly_revenue",
             labels={"month": "", "monthly_revenue": "Revenue (£)"})
fig.add_scatter(x=trend["month"], y=trend["running_total"], name="Cumulative",
                yaxis="y2", mode="lines")
fig.update_layout(
    yaxis2=dict(overlaying="y", side="right", title="Cumulative (£)"),
    showlegend=False, height=380, margin=dict(t=20, b=0),
)
st.plotly_chart(fig, width='stretch')
st.caption(
    "December 2011 is a PARTIAL month - the dataset ends on 2011-12-09. "
    "Its lower bar is truncation, not a sales decline."
)

left, right = st.columns(2)

with left:
    st.subheader("Top 10 markets")
    by_country = run_query(
        f"""
        SELECT Country, SUM(Revenue) AS revenue
        FROM sales {where}
        GROUP BY Country ORDER BY revenue DESC LIMIT 10
        """,
        params,
    )
    st.plotly_chart(
        px.bar(by_country.sort_values("revenue"), x="revenue", y="Country",
               orientation="h", labels={"revenue": "Revenue (£)", "Country": ""})
        .update_layout(height=380, margin=dict(t=20, b=0)),
        width='stretch',
    )

with right:
    st.subheader("Top 10 products")
    by_product = run_query(
        f"""
        SELECT Description AS product, SUM(Revenue) AS revenue
        FROM sales {where}
        AND Description IS NOT NULL
        GROUP BY product ORDER BY revenue DESC LIMIT 10
        """,
        params,
    )
    st.plotly_chart(
        px.bar(by_product.sort_values("revenue"), x="revenue", y="product",
               orientation="h", labels={"revenue": "Revenue (£)", "product": ""})
        .update_layout(height=380, margin=dict(t=20, b=0)),
        width='stretch',
    )
    st.caption(
        "'POSTAGE', 'Manual' and 'CARRIAGE' are service/adjustment lines, not "
        "products - see README limitations."
    )

st.subheader("Month-over-month and year-over-year growth")
growth = run_query(
    f"""
    WITH monthly AS (
        SELECT DATE_TRUNC('month', InvoiceDate) AS month,
               SUM(Revenue) AS monthly_revenue
        FROM sales {where}
        GROUP BY 1
    )
    SELECT month, monthly_revenue,
           ROUND(100.0 * (monthly_revenue - LAG(monthly_revenue, 1) OVER (ORDER BY month))
                 / LAG(monthly_revenue, 1) OVER (ORDER BY month), 1) AS mom_pct,
           ROUND(100.0 * (monthly_revenue - LAG(monthly_revenue, 12) OVER (ORDER BY month))
                 / LAG(monthly_revenue, 12) OVER (ORDER BY month), 1) AS yoy_pct
    FROM monthly ORDER BY month DESC
    """,
    params,
)
growth["month"] = pd.to_datetime(growth["month"]).dt.strftime("%Y-%m")
st.dataframe(
    growth.rename(columns={"month": "Month", "monthly_revenue": "Revenue (£)",
                           "mom_pct": "MoM %", "yoy_pct": "YoY %"}),
    width='stretch', hide_index=True, height=300,
)

st.subheader("Transaction detail")
search = st.text_input("Filter by product description (blank = all)", "")
detail_sql = f"""
    SELECT InvoiceDate, Invoice, Description, Country, Quantity, Price, Revenue
    FROM sales {where}
"""
detail_params = list(params)
if search:
    detail_sql += " AND lower(Description) LIKE ?"
    detail_params.append(f"%{search.lower()}%")
detail_sql += " ORDER BY InvoiceDate DESC LIMIT 500"

st.dataframe(run_query(detail_sql, tuple(detail_params)),
             width='stretch', hide_index=True, height=320)
st.caption("Showing up to 500 most recent matching rows.")
