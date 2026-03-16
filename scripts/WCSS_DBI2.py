import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import davies_bouldin_score, silhouette_score

def evaluate_clusters(data_csv, k_range=range(2, 8)):
    df = pd.read_csv("/Users/bazam/dev/Italian_analysis/data/datacombined-italy2.csv")
    respondent_ids = df["respondent_id"]
    df = df.drop(columns=["respondent_id"])

    # ── Explicitly define numeric score columns ──────────────────────────────
    numeric_cols = ["knowledge_score", "attitude_score", "practice_score", "risk_score"]

    # ── Everything else is categorical ──────────────────────────────────────
    # (Q columns are Likert-scale ordinal; gender/age/country/edu are binary)
    categorical_col_names = [col for col in df.columns if col not in numeric_cols]
    categorical_cols = [df.columns.get_loc(col) for col in categorical_col_names]

    results = {}
    for k in k_range:
        print(f"\n--- Evaluating {k} clusters ---")

        kproto = KPrototypes(n_clusters=k, random_state=42, init='Huang', n_init=10)
        clusters = kproto.fit_predict(df, categorical=categorical_cols)

        # ── Encode for silhouette/DBI ────────────────────────────────────────
        df_metric = df.copy()
        for col in categorical_col_names:
            df_metric[col] = df_metric[col].astype(str)
        df_encoded = pd.get_dummies(df_metric, drop_first=False)

        sil = silhouette_score(df_encoded, clusters, metric="euclidean")
        dbi = davies_bouldin_score(df_encoded, clusters)
        inertia = kproto.cost_

        results[k] = {
            "Silhouette Score": sil,
            "Inertia (WCSS)": inertia,
            "Davies-Bouldin Index": dbi,
        }
        print(f"Silhouette Score:      {sil:.3f}")
        print(f"Inertia (WCSS):        {inertia:.2f}")
        print(f"Davies-Bouldin Index:  {dbi:.3f}")

    return pd.DataFrame(results).T

# Run evaluation
results = evaluate_clusters("/Users/bazam/dev/Italian_analysis/data/datacombined-italy2.csv", k_range=range(2,8)) 
print("\n=== Clustering Evaluation Results ===") 
print(results)