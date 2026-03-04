import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/Users/bazam/dev/Italian_analysis/data/datacombined-italy.csv")
df["Cluster"] = df["Cluster"].astype(str)

clusters = sorted(df["Cluster"].unique(), key=int)

plt.figure()

for i, cluster in enumerate(clusters):
    cluster_data = df[df["Cluster"] == cluster]["practice_score"]
    
    # Violin (full, shifted slightly left)
    parts = plt.violinplot(cluster_data, positions=[i + 1 - 0.15], widths=0.25,
                           showmeans=False, showmedians=False)
    
    # Boxplot (centered)
    plt.boxplot(cluster_data, positions=[i + 1], widths=0.15,
                patch_artist=False)
    
    # Jittered scatter (right side)
    jitter = np.random.normal(i + 1 + 0.15, 0.02, size=len(cluster_data))
    plt.scatter(jitter, cluster_data, alpha=0.4)

plt.xticks(range(1, len(clusters) + 1), clusters)
plt.xlabel("Cluster")
plt.ylabel("Practice Score")
plt.title("Raincloud Plot of Practice Score per Cluster")

plt.show()