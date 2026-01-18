# Retail analytics on Online Retail II

Two years of transactions from a UK online gift retailer, loaded into DuckDB and
then pushed at from three directions: SQL window-function analysis behind a
Streamlit dashboard, a weekly revenue forecast, and an RFM customer
segmentation. Everything runs off one cleaned table.

I picked this dataset because it is messy in ways that force you to make
decisions rather than just run `groupby`. Cancellations aren't flagged with a
status column, a quarter of the rows have no customer attached, and the
best-selling "product" in most countries turns out not to be a product.

## What's in here

```
build_db.py         Excel -> cleaned DuckDB table
app.py              Streamlit dashboard
sql/01..03          the three window-function analyses
forecasting.py      Prophet vs a seasonal-naive baseline
segmentation.py     RFM + K-means
spark_aggregate.py  the monthly aggregation redone in PySpark, as a cross-check
```

## The data, and what I dropped

UCI Online Retail II, <https://archive.ics.uci.edu/ml/datasets/online+retail+II>,
CC BY 4.0. One UK non-store gift retailer, 01 Dec 2009 to 09 Dec 2011, shipped
as two Excel sheets with an identical schema, so `build_db.py` stacks them.

Out of 1,067,371 raw rows I kept 1,007,913, which is 94.4%:

| Rule | Rows dropped |
|---|---:|
| Invoice starts with `C` (cancellation) | 19,494 |
| Quantity <= 0 | 3,457 |
| Price <= 0 | 6,207 |
| Exact duplicate rows | 33,757 |

The cancellations are the one that would quietly ruin the numbers. There's no
status column, so a cancelled order is only identifiable by its invoice number,
and it carries negative quantities. Leave them in and they net off real revenue
without anything looking broken.

The decision I'd actually defend, though, is the one where I dropped nothing.
228,488 rows, about 23% of the data, have no `CustomerID`. Those are real sales,
and removing them would understate revenue by roughly a quarter, so the
dashboard and the SQL keep them. `segmentation.py` filters them out for its own
use, because you can't compute a recency for a customer that doesn't exist. Same
table, opposite rule, because the question changed.

Cleaned total revenue is £20,476,260.45 across 40,077 orders, average order
value £510.92.

## The three SQL questions

Each one is a separate file under `sql/`, written to lean on window functions
rather than pulling the data into pandas and doing it there.

**Is revenue actually growing?** `02_growth_lag.sql` puts `LAG(x, 1)` next to
`LAG(x, 12)` so month-over-month and year-over-year sit in the same row. November
2011 is up 30.6% on October, which looks excellent until the YoY column says
+2.7%. The jump is the Christmas run-up, not growth. Comparing like for like,
January to November 2011 against the same span in 2010, gives £9,182,868 against
£9,011,648, so +1.9%. A single seasonal number would have been read as a strong
year, and it isn't one.

**How concentrated is the risk?** The UK is 85.0% of revenue. Ireland, the
second market, is 3.2%. That is less a geographic spread than a single market
with a rounding error attached.

**What sells where?** `03_top_products_by_country.sql` ranks products inside each
country with `RANK() OVER (PARTITION BY Country ORDER BY product_revenue DESC)`,
and divides by a `SUM(...) OVER (PARTITION BY Country)` to get each product's
share of its own market. I expected regional taste differences. What I got was
that `POSTAGE` is the top revenue line in four of the eight biggest markets:
Germany 9.07%, Spain 8.24%, France 6.96%, Switzerland 6.62%. Add the UK, where
`DOTCOM POSTAGE` ranks first at 1.78%, and Ireland, where an accounting line
called `Manual` ranks first, and only Australia and the Netherlands have a real
product at the top.

Two readings of that. The charitable one is that shipping to continental Europe
genuinely costs a lot relative to basket size. The less charitable one is that
these are pseudo-products polluting the catalogue and any product analysis has
to exclude them by hand. Both are worth saying out loud, and I do the excluding
in the dashboard caption rather than silently in the SQL.

## The same aggregation, in PySpark

`spark_aggregate.py` recomputes revenue by country and month through the Spark
DataFrame API and then checks the result against DuckDB. Both engines return
£20,476,260.45 over 573 country-months.

This is not here because a million rows need a cluster. It's here because the
translation between SQL and the DataFrame API is the thing worth being able to
do, and having the two totals agree is the only way to know I did it right. It
runs in local mode on one JVM, which is the same API and planner, just not
distributed.

One wart: `monthly.write.parquet()` goes through the Hadoop FileSystem API,
which wants `winutils.exe` on Windows. I didn't want to install an unsigned
binary for it, so the 573 result rows come back to the driver and pandas writes
them. Spark still does all the scanning and aggregating. The native line is in
the file, commented, and works on Linux.

## Forecasting weekly revenue

Weekly rather than daily, because the retailer is shut on Saturdays. One trading
Saturday in two years means a daily series has a structural zero every week that
is noise, not signal.

The first and last weeks are partial, since the data begins and ends mid-week.
Left in, the series looks like it starts low and falls off a cliff, and the
model learns that shape. Dropping them leaves 102 complete weeks: 89 for
training, the last 13 held out.

The split is by time, never random. A random split would put future weeks into
training and score the model on information it wouldn't have had.

| Model | MAE | MAPE |
|---|---:|---:|
| Seasonal naive (52 week lag) | £54,354 | 19.6% |
| Prophet | £43,100 | 16.6% |

Prophet is 20.7% better on MAE. I want to be careful about how much credit that
deserves. On the week the buyers care about, the 14 Nov 2011 peak, actual
revenue was £387,065; the naive baseline said £380,781, missing by 1.6%, and
Prophet said £346,217, missing by 10.5%. Prophet is better on average and worse
exactly where a buyer would care. If the question is "how much stock for peak
week", the baseline is the better answer, and I'd say so.

Prophet also warns that yearly seasonality is under-identified with under two
years of history. It is fitting one-and-a-bit annual cycles. That is a real
limitation, not a formality.

## Customer segmentation

RFM on the 5,878 customers who have an identifier. Recency counts back from the
last date in the data plus a day, not from today, or every customer would look
equally stale.

Frequency and monetary get a `log1p` before scaling. A handful of trade buyers
spend orders of magnitude more than everyone else, and since K-means minimises
squared distance those few would drag the centres to themselves and leave
everyone else in one undifferentiated blob. The scaling is a separate fix for a
separate problem: monetary runs into six figures while frequency is single
digits, so without it monetary decides the distance on its own.

I chose k by silhouette but only considered k >= 3. k=2 scores best at 0.419, and
it gets there by cutting the customer base in half, which is tidy and useless:
you can't say "at risk" separately from "dormant" with two groups. k=3 scores
0.401.

| Segment | Customers | Share of base | Share of revenue | Median recency | Median orders | Median spend |
|---|---:|---:|---:|---:|---:|---:|
| Champions | 1,675 | 28.5% | 82.5% | 26 days | 11 | £3,939 |
| New / low engagement | 2,364 | 40.2% | 11.6% | 62 days | 3 | £717 |
| Dormant | 1,839 | 31.3% | 5.9% | 438 days | 1 | £326 |

28.5% of identified customers produce 82.5% of identified revenue. Note the
qualifier: this is revenue from customers we can name, which excludes that 23%
of transactions with no ID. The concentration is real, the exact percentage
isn't a statement about the whole business.

Naming the segments took two attempts. My first version ranked the clusters
against each other and split at the halfway rank, which falls apart on odd k.
With three clusters only the single most-recent one counted as recent, so a
group that had last bought 62 days ago, against an overall median of 96, came
out labelled dormant. Comparing each cluster to the overall medians instead
fixed it and holds up for any k.

I also spent longer than I'd like to admit on a reproducibility bug. Cluster
sizes moved by about 20 customers between runs even with `random_state` pinned.
The RFM query had a `GROUP BY` with no `ORDER BY`, so DuckDB's parallel
aggregate handed back rows in whatever order it finished in, and k-means++ seeds
off row order. Adding `ORDER BY 1` made it deterministic.

## What I'd tell the business

Growth is flat, roughly 2% like for like, and the headline numbers hide that
behind seasonality. Any monthly report here needs a YoY column or it will be
misread.

The customer base has a retention problem rather than an acquisition problem.
31.3% of identified customers are dormant at a median 438 days, and the 1,675
Champions carrying 82.5% of revenue are the obvious thing to protect first.

Shipping is worth a proper look. When postage is the single largest revenue line
in four European markets, either it's priced above what the basket justifies, or
those markets are being served inefficiently. Both are answerable with data the
business already has.

For peak-week stock planning I'd use the seasonal-naive number, not Prophet,
for the reason in the forecasting section.

## Running it

```bash
pip install -r requirements-dev.txt

# the warehouse is committed, so the dashboard runs straight away
streamlit run app.py

python forecasting.py
python segmentation.py
```

To rebuild the warehouse from scratch, put `online_retail_II.xlsx` in `data/raw/`
(45 MB, link above) and run `python build_db.py`. Takes about three minutes,
nearly all of it parsing Excel. The raw file is gitignored since it is freely
downloadable; `requirements.txt` on its own covers just the dashboard, which is
what the deployed version installs.

`spark_aggregate.py` additionally needs `pyspark` and a JDK with `JAVA_HOME`
set. It's optional and nothing else depends on it.

## What I know is wrong with it

The `POSTAGE`, `Manual` and `CARRIAGE` lines are still in the base table. I
exclude them where it matters and flag them in the dashboard, but a stricter
build would separate service lines from products at load time.

Segmentation only sees 77% of transactions, because the rest have no customer.

Two years is thin for yearly seasonality, and Prophet says so itself.

The forecast is a single 13-week holdout rather than rolling-origin
cross-validation, so the accuracy figures rest on one particular quarter, and
that quarter contains the Christmas peak.

Nothing here is incremental. `build_db.py` rebuilds the whole table each run,
which is fine at a million rows and would not be at a hundred million.
