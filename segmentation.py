"""RFM segmentation of the identified customers.

Builds Recency / Frequency / Monetary per customer, clusters with K-means,
picks k from the elbow and silhouette, then gives each cluster a business name.

Run:  python segmentation.py
Writes: outputs/elbow_silhouette.png, outputs/rfm_clusters.csv,
        outputs/cluster_profiles.csv
"""

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DB = "data/retail.duckdb"
K_RANGE = range(2, 9)
RANDOM_STATE = 42


def build_rfm() -> pd.DataFrame:
    """One row per customer: recency, frequency, monetary.

    Null CustomerID rows are dropped here. You cannot compute a recency for a
    customer that does not exist, so about a quarter of the transactions take
    no part in this analysis at all.

    Recency counts back from the last date in the data plus a day, not from
    today. The data stops in Dec 2011, so today would just make every customer
    look equally stale.

    ORDER BY matters more than it looks. Without it DuckDB returns the groups
    in whatever order the parallel aggregate finishes in, and since k-means++
    seeds from the row order the cluster sizes shifted by ~20 customers between
    runs even with random_state pinned.
    """
    con = duckdb.connect(DB, read_only=True)
    df = con.execute(
        """
        WITH ref AS (SELECT MAX(InvoiceDate) + INTERVAL 1 DAY AS as_of FROM sales)
        SELECT
            CAST(s.CustomerID AS BIGINT)                              AS customer_id,
            DATE_DIFF('day', MAX(s.InvoiceDate), (SELECT as_of FROM ref)) AS recency_days,
            COUNT(DISTINCT s.Invoice)                                 AS frequency,
            SUM(s.Revenue)                                            AS monetary
        FROM sales s
        WHERE s.CustomerID IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
    con.close()
    return df


def prepare_features(rfm: pd.DataFrame) -> np.ndarray:
    """Log-transforms then standardises the three RFM columns.

    Two steps for two different reasons. The log is for skew: a handful of
    trade buyers spend orders of magnitude more than everyone else, and since
    K-means minimises squared distance those few would drag the centres toward
    themselves and leave the other 5,800 customers in one blob.

    The scaling is for units. Monetary runs into the hundreds of thousands
    while frequency is single digits, so without it monetary decides the
    distance on its own and frequency may as well not be there.
    """
    features = rfm[["recency_days", "frequency", "monetary"]].copy()
    features["frequency"] = np.log1p(features["frequency"])
    features["monetary"] = np.log1p(features["monetary"])
    return StandardScaler().fit_transform(features)


def evaluate_k(X: np.ndarray) -> pd.DataFrame:
    """Fits K-means across K_RANGE, recording inertia and silhouette.

    Inertia always falls as k rises, so on its own it cannot choose k -- you
    are looking for the elbow. Silhouette does not automatically improve with
    more clusters, so it can.
    """
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels),
        })
    return pd.DataFrame(rows)


def plot_selection(scores: pd.DataFrame, chosen_k: int) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(scores["k"], scores["inertia"], marker="o", color="#0077b6")
    ax1.set_title("Elbow method")
    ax1.set_xlabel("k"); ax1.set_ylabel("Inertia")
    ax2.plot(scores["k"], scores["silhouette"], marker="o", color="#d1495b")
    ax2.set_title("Silhouette score")
    ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette")
    for ax in (ax1, ax2):
        ax.axvline(chosen_k, ls=":", color="#888")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("outputs/elbow_silhouette.png", dpi=140)


def name_personas(profile: pd.DataFrame, rfm: pd.DataFrame) -> dict[int, str]:
    """Labels each cluster by where it sits against the overall medians.

    I compare to the medians rather than ranking the clusters against each
    other. My first version ranked them and split at the halfway rank, which
    falls apart on odd k -- with three clusters only the single most-recent one
    counted as recent, so a cluster that had last bought 62 days ago, against a
    median of 96, came out labelled dormant.
    """
    med_r = rfm["recency_days"].median()
    med_f = rfm["frequency"].median()
    med_m = rfm["monetary"].median()

    names = {}
    for cid, row in profile.iterrows():
        recent = row["recency_days"] <= med_r         # lower recency = better
        valuable = (row["frequency"] >= med_f) and (row["monetary"] >= med_m)
        if recent and valuable:
            names[cid] = "Champions - recent and high value"
        elif recent and not valuable:
            names[cid] = "New / low-engagement - recent but small"
        elif not recent and valuable:
            names[cid] = "At risk - valuable but slipping away"
        else:
            names[cid] = "Dormant - inactive and low value"
    return names


def main() -> None:
    rfm = build_rfm()
    print(f"customers with an identifier: {len(rfm):,}")
    print(rfm[["recency_days", "frequency", "monetary"]]
          .describe(percentiles=[.5, .9, .99])
          .to_string(float_format=lambda v: f"{v:,.1f}"))

    X = prepare_features(rfm)

    scores = evaluate_k(X)
    print("\n--- choosing k ---")
    print(scores.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    # Best silhouette, but only from k >= 3. k=2 tends to win by cutting the
    # data in half, which scores well and says nothing useful: you cannot
    # express "at risk" separately from "dormant" with two groups.
    candidates = scores[scores["k"] >= 3]
    chosen_k = int(candidates.loc[candidates["silhouette"].idxmax(), "k"])
    print(f"\nchosen k = {chosen_k} "
          f"(best silhouette among k>=3: "
          f"{candidates['silhouette'].max():.3f})")

    km = KMeans(n_clusters=chosen_k, random_state=RANDOM_STATE, n_init=10)
    rfm["cluster"] = km.fit_predict(X)

    profile = rfm.groupby("cluster").agg(
        customers=("customer_id", "count"),
        recency_days=("recency_days", "median"),
        frequency=("frequency", "median"),
        monetary=("monetary", "median"),
        total_revenue=("monetary", "sum"),
    )
    profile["pct_of_customers"] = (profile["customers"] / len(rfm) * 100).round(1)
    profile["pct_of_revenue"] = (
        profile["total_revenue"] / profile["total_revenue"].sum() * 100
    ).round(1)
    profile["persona"] = pd.Series(name_personas(profile, rfm))

    print("\n--- cluster profiles (medians) ---")
    print(profile.to_string(float_format=lambda v: f"{v:,.1f}"))

    rfm.to_csv("outputs/rfm_clusters.csv", index=False)
    profile.to_csv("outputs/cluster_profiles.csv")
    plot_selection(scores, chosen_k)
    print("\nsaved outputs/elbow_silhouette.png, rfm_clusters.csv, cluster_profiles.csv")


if __name__ == "__main__":
    main()
