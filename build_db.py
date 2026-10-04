"""Loads the raw Online Retail II spreadsheets into a DuckDB warehouse.

Source: https://archive.ics.uci.edu/ml/datasets/online+retail+II (CC BY 4.0).
One UK online gift retailer, Dec 2009 to Dec 2011, split over two Excel sheets.

Run once:  python build_db.py
Produces:  data/retail.duckdb, one table called `sales`
"""

import duckdb
import pandas as pd

RAW_XLSX = "data/raw/online_retail_II.xlsx"
DB_PATH = "data/retail.duckdb"


def load_raw() -> pd.DataFrame:
    """Both sheets share a schema, so I stack them into one frame."""
    sheets = pd.read_excel(RAW_XLSX, sheet_name=None, engine="openpyxl")
    print(f"sheets found: {list(sheets)}")
    for name, frame in sheets.items():
        print(f"  {name}: {len(frame):,} rows")

    df = pd.concat(sheets.values(), ignore_index=True)

    # This release calls the column 'Customer ID', with a space, which would
    # need quoting in every SQL query. Rename it once here instead.
    df = df.rename(columns={"Customer ID": "CustomerID"})
    return df


def net_out_matched_returns(df: pd.DataFrame, cancelled: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drops the original sale row behind a cancellation, not just the cancellation.

    Dropping cancelled (negative-quantity) rows alone is not enough: if a
    customer bought 5 units and later returned all 5, the original +5 sale
    row is still sitting in the data counting real revenue for a product the
    customer ended up not keeping. Only dropping the cancellation and leaving
    the original sale overstates revenue by the full value of every matched
    return.

    This matches each cancellation to the earliest not-yet-matched sale row
    with the same CustomerID, StockCode and Quantity (magnitude), dated on or
    before the cancellation (FIFO, one sale consumed per cancellation) and
    drops that original sale row too. Cancellations with no CustomerID, or no
    matching prior sale in this dataset (the purchase may predate this
    extract, or be a partial/mismatched-quantity return), can't be matched
    and are left as an unmatched drop of just the negative row -- which is
    revenue-neutral, since the matching sale (if any) was never counted here
    in the first place.
    """
    cancelled = cancelled.copy()
    cancelled["AbsQty"] = cancelled["Quantity"].abs()
    cancelled_valid = cancelled.dropna(subset=["CustomerID"]).sort_values("InvoiceDate")

    sales_idx = df.dropna(subset=["CustomerID"]).sort_values("InvoiceDate")
    groups: dict[tuple, list[int]] = {}
    for key, g in sales_idx.groupby(["CustomerID", "StockCode", "Quantity"]):
        groups[key] = list(g.index)
    date_lookup = sales_idx["InvoiceDate"]

    to_drop: set[int] = set()
    for _, row in cancelled_valid.iterrows():
        key = (row["CustomerID"], row["StockCode"], row["AbsQty"])
        idx_list = groups.get(key)
        if not idx_list:
            continue
        while idx_list and idx_list[0] in to_drop:
            idx_list.pop(0)
        if idx_list and date_lookup.loc[idx_list[0]] <= row["InvoiceDate"]:
            to_drop.add(idx_list.pop(0))

    return df.drop(index=to_drop), len(to_drop)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Returns the cleaned frame plus a count of what each rule dropped."""
    report = {"rows_in": len(df)}

    # There's no status column; a cancelled order is just an Invoice starting
    # with 'C'. They carry negative Quantity, so leaving them in nets off real
    # revenue without any warning.
    df["Invoice"] = df["Invoice"].astype(str)
    is_cancelled = df["Invoice"].str.startswith("C")
    cancelled = df[is_cancelled]
    report["cancelled"] = int(is_cancelled.sum())
    df = df[~is_cancelled]

    # Returns, freebies and adjustments that were never flagged as cancellations.
    bad_qty = df["Quantity"] <= 0
    bad_price = df["Price"] <= 0
    report["non_positive_qty"] = int(bad_qty.sum())
    report["non_positive_price"] = int(bad_price.sum())
    df = df[~(bad_qty | bad_price)]

    # Same invoice, product, quantity and timestamp.
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates"] = before - len(df)

    # The cancellation rows are gone, but the sale they reversed is still
    # sitting in df counting revenue it shouldn't. Find and drop that
    # matching original sale too -- see net_out_matched_returns for why.
    df, matched_returns = net_out_matched_returns(df, cancelled)
    report["matched_returns_netted"] = matched_returns

    # Defined once here so every query downstream means the same thing by it.
    df["Revenue"] = df["Quantity"] * df["Price"]

    # I keep rows with a null CustomerID. They're about a quarter of the data
    # and they're real sales, so dropping them here would understate revenue.
    # segmentation.py filters them out for its own use, because you can't
    # compute a recency for a customer that doesn't exist.
    report["null_customer_id"] = int(df["CustomerID"].isna().sum())
    report["rows_out"] = len(df)
    return df, report


def main() -> None:
    df = load_raw()
    df, report = clean(df)

    print("\n--- cleaning report ---")
    for key, value in report.items():
        print(f"{key:>20}: {value:,}")
    kept = report["rows_out"] / report["rows_in"] * 100
    print(f"{'kept':>20}: {kept:.1f}% of raw rows")

    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS sales")
    # DuckDB picks the pandas frame straight out of local scope.
    con.execute("CREATE TABLE sales AS SELECT * FROM df")
    con.execute("CREATE INDEX idx_sales_date ON sales(InvoiceDate)")

    n, lo, hi, rev = con.execute(
        "SELECT COUNT(*), MIN(InvoiceDate), MAX(InvoiceDate), SUM(Revenue) FROM sales"
    ).fetchone()
    print(f"\nwrote {n:,} rows to {DB_PATH}")
    print(f"date range : {lo:%Y-%m-%d} -> {hi:%Y-%m-%d}")
    print(f"revenue    : GBP {rev:,.0f}")
    con.close()


if __name__ == "__main__":
    main()
