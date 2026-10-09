import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH = "/Users/bazam/dev/Italian_analysis/data/datacombined-italy2.csv"
NUMERIC_COLS = ["knowledge_score", "attitude_score", "practice_score", "risk_score"]
N_BOOTSTRAP = 50       # number of bootstrap runs
SUBSAMPLE_RATIO = 0.8  # fraction of data per bootstrap run

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_and_prep(data_csv):
    df = pd.read_csv(data_csv)
    respondent_ids = df["respondent_id"]
    df = df.drop(columns=["respondent_id"])
    categorical_col_names = [col for col in df.columns if col not in NUMERIC_COLS]
    categorical_cols = [df.columns.get_loc(col) for col in categorical_col_names]
    return df, respondent_ids, categorical_col_names, categorical_cols


def encode_for_metrics(df, categorical_col_names):
    df_for_sil = df.copy()
    for col in categorical_col_names:
        df_for_sil[col] = df_for_sil[col].astype(str)
    return pd.get_dummies(df_for_sil, drop_first=False)


def fit_kproto(df, categorical_cols, k):
    kproto = KPrototypes(n_clusters=k, random_state=42, init='Huang', n_init=10)
    clusters = kproto.fit_predict(df, categorical=categorical_cols)
    return clusters


# ── Elbow method: K-Prototypes cost ─────────────────────────────────────────

def elbow_analysis(df, categorical_cols, k_range=range(1, 9)):
    """
    Calculate K-Prototypes cost for each candidate k.
    Lower cost indicates a better fit, but cost always tends
    to decrease as the number of clusters increases.
    """
    results = []

    for k in k_range:
        model = KPrototypes(
            n_clusters=k,
            random_state=42,
            init="Huang",
            n_init=10
        )

        model.fit_predict(df, categorical=categorical_cols)

        results.append({
            "k": k,
            "Cost": model.cost_
        })

        print(f"k = {k}: cost = {model.cost_:.2f}")

    elbow_df = pd.DataFrame(results)

    # Plot elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(
        elbow_df["k"],
        elbow_df["Cost"],
        marker="o"
    )
    plt.xticks(elbow_df["k"])
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("K-Prototypes cost")
    plt.title("Elbow Method for K-Prototypes")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    plt.savefig(
        "/Users/bazam/dev/Italian_analysis/results/kprototypes_elbow.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    return elbow_df

# ── 1. Per-cluster silhouette breakdown ──────────────────────────────────────

def per_cluster_silhouette(df, categorical_col_names, categorical_cols, k):
    """Returns mean silhouette per cluster, not just the global average."""
    clusters = fit_kproto(df, categorical_cols, k)
    df_encoded = encode_for_metrics(df, categorical_col_names)
    
    sample_sils = silhouette_samples(df_encoded, clusters, metric="euclidean")
    
    cluster_sils = {}
    for c in sorted(set(clusters)):
        mask = clusters == c
        cluster_sils[c] = {
            "mean_silhouette": sample_sils[mask].mean(),
            "n_members": mask.sum()
        }
    return cluster_sils, sample_sils, clusters


# ── 2. Bootstrap stability (ARI-based) ───────────────────────────────────────

def bootstrap_stability(df, categorical_col_names, categorical_cols, k,
                        n_bootstrap=N_BOOTSTRAP, subsample_ratio=SUBSAMPLE_RATIO):
    """
    For each bootstrap iteration:
      - Subsample the data
      - Cluster it
      - Compare to full-data clustering via Adjusted Rand Index (ARI)
    Returns mean ARI across iterations (1.0 = perfectly stable, 0.0 = random).
    """
    # Full-data clustering as reference
    full_clusters = fit_kproto(df, categorical_cols, k)
    
    ari_scores = []
    n = len(df)
    subsample_size = int(n * subsample_ratio)
    
    for i in range(n_bootstrap):
        idx = np.random.choice(n, size=subsample_size, replace=False)
        df_sub = df.iloc[idx].reset_index(drop=True)
        
        sub_clusters = fit_kproto(df_sub, categorical_cols, k)
        
        # ARI between full-data labels (on same subset) and bootstrap labels
        ari = adjusted_rand_score(full_clusters[idx], sub_clusters)
        ari_scores.append(ari)
    
    return np.mean(ari_scores), np.std(ari_scores)


# ── 3. Cronbach's alpha within each cluster ───────────────────────────────────

def cronbach_alpha(df_subset):
    """Cronbach's alpha for a numeric dataframe subset."""
    df_subset = df_subset.select_dtypes(include=[np.number])
    n_items = df_subset.shape[1]
    if n_items < 2:
        return np.nan
    item_variances = df_subset.var(axis=0, ddof=1).sum()
    total_variance = df_subset.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return np.nan
    alpha = (n_items / (n_items - 1)) * (1 - item_variances / total_variance)
    return alpha


def intracluster_cronbach(df, categorical_col_names, categorical_cols, k):
    """Compute Cronbach's alpha on numeric columns within each cluster."""
    clusters = fit_kproto(df, categorical_cols, k)
    df_with_clusters = df.copy()
    df_with_clusters["Cluster"] = clusters
    
    results = {}
    for c in sorted(set(clusters)):
        subset = df_with_clusters[df_with_clusters["Cluster"] == c][NUMERIC_COLS]
        results[c] = cronbach_alpha(subset)
    return results


# ── Run everything ────────────────────────────────────────────────────────────

df, respondent_ids, categorical_col_names, categorical_cols = load_and_prep(DATA_PATH)

# ── Elbow analysis ──────────────────────────────────────────────────────────
elbow_df = elbow_analysis(
    df,
    categorical_cols,
    k_range=range(1, 9)
)

print("\n=== Elbow Method Results ===")
print(elbow_df.round(2))


from sklearn.metrics import adjusted_rand_score

# ── Compare membership: k=2 versus k=4 ──────────────────────────────────────

# Fit both solutions to the full dataset
clusters_k2 = fit_kproto(df, categorical_cols, 2)
clusters_k4 = fit_kproto(df, categorical_cols, 4)

comparison = pd.DataFrame({
    "k2_cluster": clusters_k2,
    "k4_cluster": clusters_k4
})

# Cross-tabulation of cluster membership
cross_tab = pd.crosstab(
    comparison["k2_cluster"],
    comparison["k4_cluster"],
    margins=True
)

print("\n=== Cluster Membership Cross-tabulation ===")
print(cross_tab)

# Identify the smaller cluster in the k=2 solution
sizes_k2 = pd.Series(clusters_k2).value_counts()
small_cluster_k2 = sizes_k2.idxmin()

small_group = comparison[
    comparison["k2_cluster"] == small_cluster_k2
]

n_small_k2 = len(small_group)

# Assess overlap with each k=4 cluster
overlap_results = []

for cluster4 in sorted(comparison["k4_cluster"].unique()):
    group4 = comparison[comparison["k4_cluster"] == cluster4]

    intersection = len(
        small_group[small_group["k4_cluster"] == cluster4]
    )

    n_cluster4 = len(group4)
    union = n_small_k2 + n_cluster4 - intersection

    overlap_results.append({
        "k2_small_cluster": small_cluster_k2,
        "k4_cluster": cluster4,
        "n_k2_small": n_small_k2,
        "n_k4_cluster": n_cluster4,
        "overlap_n": intersection,
        "k2_cluster_retained_%": 100 * intersection / n_small_k2,
        "k4_cluster_purity_%": 100 * intersection / n_cluster4,
        "Jaccard_similarity": intersection / union
    })

overlap_df = pd.DataFrame(overlap_results).sort_values(
    "Jaccard_similarity", ascending=False
)

print("\n=== Overlap of the Smaller k=2 Cluster with k=4 Clusters ===")
print(overlap_df.round(3).to_string(index=False))

# Overall agreement between the two partitions
overall_ari = adjusted_rand_score(clusters_k2, clusters_k4)

print(f"\nOverall ARI between k=2 and k=4: {overall_ari:.3f}")

summary = {}

for k in range(2, 6):
    print(f"\n{'='*40}\n k = {k} clusters\n{'='*40}")
    
    # Global silhouette
    clusters = fit_kproto(df, categorical_cols, k)
    df_encoded = encode_for_metrics(df, categorical_col_names)
    global_sil = silhouette_score(df_encoded, clusters, metric="euclidean")
    
    # Per-cluster silhouette
    cluster_sils, _, _ = per_cluster_silhouette(df, categorical_col_names, categorical_cols, k)
    sil_std = np.std([v["mean_silhouette"] for v in cluster_sils.values()])
    
    # Bootstrap stability
    mean_ari, std_ari = bootstrap_stability(df, categorical_col_names, categorical_cols, k)
    
    # Cronbach's alpha per cluster
    alphas = intracluster_cronbach(df, categorical_col_names, categorical_cols, k)
    mean_alpha = np.nanmean(list(alphas.values()))
    
    print(f"  Global Silhouette:        {global_sil:.3f}")
    print(f"  Silhouette std (clusters):{sil_std:.3f}  ← lower = more even clusters")
    print(f"  Bootstrap Stability (ARI):{mean_ari:.3f} ± {std_ari:.3f}  ← higher = more stable")
    print(f"  Mean Cronbach's α:        {mean_alpha:.3f}  ← >0.7 = internally consistent")
    for c, a in alphas.items():
        print(f"    Cluster {c}: α = {a:.3f}  (n={sum(clusters==c)})")
    
    summary[k] = {
        "Global Silhouette": global_sil,
        "Silhouette Std": sil_std,
        "Bootstrap Stability (ARI)": mean_ari,
        "ARI Std": std_ari,
        "Mean Cronbach Alpha": mean_alpha,
    }

# ── Summary table ─────────────────────────────────────────────────────────────
summary_df = pd.DataFrame(summary).T
print("\n=== Full Summary ===")
print(summary_df.round(3))


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ks = list(summary.keys())

axes[0].plot(ks, summary_df["Global Silhouette"], marker="o", color="teal")
axes[0].set_title("Silhouette Score"); axes[0].set_xlabel("k"); axes[0].grid(True, alpha=0.4)

axes[1].errorbar(ks, summary_df["Bootstrap Stability (ARI)"],
                 yerr=summary_df["ARI Std"], marker="o", color="steelblue", capsize=4)
axes[1].set_title("Bootstrap Stability (ARI)"); axes[1].set_xlabel("k"); axes[1].grid(True, alpha=0.4)

axes[2].plot(ks, summary_df["Mean Cronbach Alpha"], marker="o", color="coral")
axes[2].axhline(0.7, linestyle="--", color="gray", label="α = 0.7 threshold")
axes[2].set_title("Mean Cronbach's α per Cluster"); axes[2].set_xlabel("k")
axes[2].legend(); axes[2].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("/Users/bazam/dev/Italian_analysis/results/internal_consistency.png", dpi=300)
plt.show()