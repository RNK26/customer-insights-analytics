"""The monthly-revenue-by-country aggregation again, this time in PySpark.

sql/01_monthly_revenue_trend.sql does this against DuckDB. This does the same
work through the Spark DataFrame API so the two can be read side by side, and
finishes by checking the totals agree.

Spark runs in local mode here -- one JVM on this machine, no cluster. Same API
and same query planner, just not distributed.

Run:  python spark_aggregate.py
Writes: outputs/spark/revenue_by_country_month.parquet
"""

import os
import shutil

import duckdb
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DB = "data/retail.duckdb"
STAGING_PARQUET = "outputs/spark/sales_staging.parquet"
OUT_PARQUET = "outputs/spark/revenue_by_country_month.parquet"


def export_from_duckdb() -> None:
    """Spark cannot read a .duckdb file, so hand the data over as Parquet."""
    os.makedirs("outputs/spark", exist_ok=True)
    con = duckdb.connect(DB, read_only=True)
    con.execute(f"COPY (SELECT * FROM sales) TO '{STAGING_PARQUET}' (FORMAT PARQUET)")
    n = con.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
    con.close()
    size_mb = os.path.getsize(STAGING_PARQUET) / 1e6
    print(f"exported {n:,} rows to Parquet ({size_mb:.1f} MB)")


def main() -> None:
    export_from_duckdb()

    spark = (
        SparkSession.builder
        .appName("retail-monthly-revenue")
        .master("local[*]")
        # default is 200 partitions, which is far too many for 1M rows on one box
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print(f"Spark {spark.version} | master: {spark.sparkContext.master}")

    df = spark.read.parquet(STAGING_PARQUET)

    monthly = (
        df
        .withColumn("month", F.date_trunc("month", F.col("InvoiceDate")))
        .groupBy("Country", "month")
        .agg(
            F.sum("Revenue").alias("revenue"),
            F.countDistinct("Invoice").alias("orders"),
            F.sum("Quantity").alias("units"),
        )
        .withColumn("avg_order_value", F.col("revenue") / F.col("orders"))
        .orderBy(F.col("revenue").desc())
    )

    # Nothing above has touched the data yet. show() is the first action, and
    # that is what makes Spark actually run the whole chain.
    print("\n--- top 10 country-months by revenue ---")
    monthly.show(10, truncate=False)
    print(f"aggregated rows: {monthly.count():,}")

    # monthly.write.parquet() goes through the Hadoop FileSystem API, which
    # wants winutils.exe and a HADOOP_HOME on Windows. I would rather not
    # install an unsigned binary for this, so the 573 result rows come back to
    # the driver and pandas writes them. Spark still does the scan and the
    # aggregation; only the final write is local. On Linux or Databricks the
    # commented line works as-is.
    #
    # monthly.write.parquet(OUT_PARQUET)
    if os.path.exists(OUT_PARQUET):
        shutil.rmtree(OUT_PARQUET) if os.path.isdir(OUT_PARQUET) else os.remove(OUT_PARQUET)
    monthly.toPandas().to_parquet(OUT_PARQUET)
    print(f"wrote {OUT_PARQUET}")

    # If the two engines disagree the translation is wrong somewhere.
    con = duckdb.connect(DB, read_only=True)
    duck_total = con.execute("SELECT SUM(Revenue) FROM sales").fetchone()[0]
    con.close()
    spark_total = df.agg(F.sum("Revenue")).collect()[0][0]
    print(f"\nrevenue check  DuckDB: {duck_total:,.2f}  |  Spark: {spark_total:,.2f}  "
          f"|  match: {abs(duck_total - spark_total) < 0.01}")

    spark.stop()


if __name__ == "__main__":
    main()
