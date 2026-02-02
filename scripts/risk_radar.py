import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------------
base_in = "/Users/bazam/dev/Italian_analysis/data/"
base_out = "/Users/bazam/dev/Italian_analysis/results/"

practice_file = os.path.join(base_in, "practice-italy.csv")
knowledge_file = os.path.join(base_in, "demo_clusters.csv")
attitude_file = os.path.join(base_in, "attitude-italy.csv")
risk_path = os.path.join(base_in, "risk-italy.csv")

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
df_k = pd.read_csv(knowledge_file)
df_a = pd.read_csv(attitude_file)
df_p = pd.read_csv(practice_file)
df_r = pd.read_csv(risk_path)

# -----------------------------------------------------------------------------
# Clean and compute composite risk perception
# -----------------------------------------------------------------------------
df_r = df_r.replace(0, np.nan)
risk_cols = [c for c in df_r.columns if c != "respondent_id"]
df_r["risk_composite"] = df_r[risk_cols].mean(axis=1)

# -----------------------------------------------------------------------------
# Merge datasets
# -----------------------------------------------------------------------------
df_full = (
    df_k.merge(df_a, on="respondent_id", suffixes=("", "_attitude"))
         .merge(df_p, on="respondent_id", suffixes=("", "_practice"))
         .merge(df_r[["respondent_id", "risk_composite"]], on="respondent_id", how="left")
)

# -----------------------------------------------------------------------------
# Compute composite DKAP+R scores
# -----------------------------------------------------------------------------
attitude_cols = [c for c in df_a.columns if c != "respondent_id"]
practice_cols = [c for c in df_p.columns if c != "respondent_id"]

df_full["attitude_composite"] = df_full[attitude_cols].mean(axis=1)
df_full["practice_composite"] = df_full[practice_cols].mean(axis=1)

# Final variable list (K, A, P, R)
dkapr_vars = ["knowledge_score", "attitude_composite", "practice_composite", "risk_composite"]

# -----------------------------------------------------------------------------
# Normalize (0–1) for visualization
# -----------------------------------------------------------------------------
scaler = MinMaxScaler()
normalized = df_full[["respondent_id"] + dkapr_vars].copy()
normalized[dkapr_vars] = scaler.fit_transform(normalized[dkapr_vars])

# -----------------------------------------------------------------------------
# Integrate existing knowledge-based clusters
# -----------------------------------------------------------------------------
if "Cluster" not in df_full.columns:
    df_clusters = df_k[["respondent_id", "Cluster"]]
    normalized = pd.merge(normalized, df_clusters, on="respondent_id", how="left")
else:
    normalized["Cluster"] = df_full["Cluster"]

# -----------------------------------------------------------------------------
# Radar plot by knowledge-based cluster (now DKAP + Risk)
# -----------------------------------------------------------------------------
categories = dkapr_vars
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(10, 10))
for cluster_id, subset in normalized.groupby("Cluster"):
    values = subset[categories].mean().tolist()
    values += values[:1]
    plt.polar(angles, values, label=f"Cluster {int(cluster_id)}", linewidth=2)
plt.xticks(angles[:-1], categories, color="grey", size=12)
plt.title("DKAP + Risk Profiles by Knowledge-Based Demographic Clusters", size=16, y=1.08)
plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
plt.tight_layout()

radar_path = os.path.join(base_out, "DKAPR_cluster_profiles.png")
plt.savefig(radar_path, dpi=300, bbox_inches="tight")
plt.close()

print("✅ DKAPR radar plot saved to:", radar_path)

# -----------------------------------------------------------------------------
# 📊 Radar plot with individual risk perception variables (Q6, Q15, Q17, Q26)
# -----------------------------------------------------------------------------

# Load the raw risk variables again (already cleaned in df_r)
risk_items = ["Q6", "Q15", "Q17", "Q26"]

# Merge those into the main dataset
df_risk_items = df_r[["respondent_id"] + risk_items]
df_full_risk = df_full.merge(df_risk_items, on="respondent_id", how="left")

# Define the DKAP + individual risk variable set
dkapr_items_vars = ["knowledge_score", "attitude_composite", "practice_composite"] + risk_items

# Normalize everything (0–1)
normalized_items = df_full_risk[["respondent_id"] + dkapr_items_vars].copy()
normalized_items[risk_items] = normalized_items[risk_items].replace(0, np.nan)
scaler = MinMaxScaler()
normalized_items[dkapr_items_vars] = scaler.fit_transform(normalized_items[dkapr_items_vars])

# Add existing clusters
if "Cluster" not in df_full_risk.columns:
    df_clusters = df_k[["respondent_id", "Cluster"]]
    normalized_items = pd.merge(normalized_items, df_clusters, on="respondent_id", how="left")
else:
    normalized_items["Cluster"] = df_full_risk["Cluster"]

# --- Radar parameters ---
categories = dkapr_items_vars
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(12, 12))
for cluster_id, subset in normalized_items.groupby("Cluster"):
    values = subset[categories].mean().tolist()
    values += values[:1]
    plt.polar(angles, values, label=f"Cluster {int(cluster_id)}", linewidth=2)
plt.xticks(angles[:-1], categories, color="grey", size=11)
plt.title("DKAP + Individual Risk Variables by Knowledge-Based Clusters", size=16, y=1.1)
plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
plt.tight_layout()

radar_path_items = os.path.join(base_out, "DKAPR_individual_items_cluster_profiles.png")
plt.savefig(radar_path_items, dpi=300, bbox_inches="tight")
plt.close()

print("✅ DKAPR (individual risk variables) radar plot saved to:", radar_path_items)