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


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Returns the cleaned frame plus a count of what each rule dropped."""
    report = {"rows_in": len(df)}

    # There's no status column; a cancelled order is just an Invoice starting
    # with 'C'. They carry negative Quantity, so leaving them in nets off real
    # revenue without any warning.
    df["Invoice"] = df["Invoice"].astype(str)
    is_cancelled = df["Invoice"].str.startswith("C")
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
