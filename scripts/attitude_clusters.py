import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

base_input = "/Users/bazam/dev/Italian_analysis/data/"
base_output = "/Users/bazam/dev/Italian_analysis/results/"

attitude_file = os.path.join(base_input, "attitude-italy.csv")

df_full = pd.read_csv(attitude_file)

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df_full,
    x="Cluster",
    y="attitude_composite",
    showfliers=False,
    linewidth=1.2
)

sns.stripplot(
    data=df_full,
    x="Cluster",
    y="attitude_composite",
    color="black",
    alpha=0.25,
    jitter=0.25,
    size=3
)

plt.title("Distribution of Attitude Scores across Knowledge-Based Clusters", fontsize=14)
plt.xlabel("Knowledge-Based Cluster")
plt.ylabel("Attitude Composite Score")
plt.tight_layout()

out_path = os.path.join(base_output, "attitude_distribution_by_cluster.png")
plt.savefig(out_path, dpi=300)
plt.close()