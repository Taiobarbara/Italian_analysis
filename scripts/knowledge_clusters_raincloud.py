"""
Create a raincloud-style plot of `risk_score` for each cluster.

This script reads the combined dataset, groups `risk_score` values by
cluster, and draws for each cluster a composite plot made of:
 - a violin (to show the distribution density),
 - a boxplot (to show median and IQR), and
 - jittered scatter points (to show individual observations).

The violin/box/scatter are slightly offset horizontally so they appear
as a single raincloud for each cluster.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Data loading and preparation ---
# Read the combined dataset (adjust path if repo is moved)
df = pd.read_csv("/Users/bazam/dev/Italian_analysis/data/datacombined-italy.csv")

# Ensure cluster labels are strings (helps with sorting and plotting)
df["Cluster"] = df["Cluster"].astype(str)

# Get sorted list of clusters (cast to int for numeric sort, then back to str)
clusters = sorted(df["Cluster"].unique(), key=int)

# --- Plot setup ---
plt.figure()

# For each cluster draw violin (left), boxplot (center), and scatter (right)
for i, cluster in enumerate(clusters):
    # Select the numeric scores for this cluster
    cluster_data = df[df["Cluster"] == cluster]["risk_score"]

    # Violin: show the full distribution density, shifted left slightly
    # positions: horizontal x-location; widths: overall violin width
    parts = plt.violinplot(cluster_data, positions=[i + 1 - 0.15], widths=0.25,
                           showmeans=False, showmedians=False)

    # Boxplot: show median and IQR, centered at the cluster's x-position
    plt.boxplot(cluster_data, positions=[i + 1], widths=0.15,
                patch_artist=False)

    # Jittered scatter: display individual observations to the right
    # Use a small Gaussian jitter so points overlap less and remain near x=i+1
    jitter = np.random.normal(i + 1 + 0.15, 0.02, size=len(cluster_data))
    plt.scatter(jitter, cluster_data, alpha=0.4)

# X-axis ticks and labels correspond to cluster identifiers
plt.xticks(range(1, len(clusters) + 1), clusters)
plt.xlabel("Cluster")

# Y-axis label describes what is being plotted (score variable)
plt.ylabel("Practice Score")
plt.title("Raincloud Plot of Risk Score per Cluster")

# Show the composed raincloud plot
plt.show()