import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("/Users/bazam/dev/Italian_analysis/data/datacombined-italy.csv")

risk_items = ["Q6","Q15","Q17","Q26"]

# Ensure cluster order
clusters = sorted(df["Cluster"].unique())

for q in risk_items:

    plt.figure(figsize=(8,6))

    for i, cluster in enumerate(clusters):

        data = df[df["Cluster"] == cluster][q]

        sns.kdeplot(
            data,
            bw_adjust=0.7,
            fill=True,
            alpha=0.6,
            linewidth=1.2,
            label=f"Cluster {cluster}",
            clip=(data.min(), data.max())
        )

    plt.title(f"Ridgeline Distribution of {q} by Cluster")
    plt.xlabel("Response Level")
    plt.ylabel("Density")
    plt.legend(title="Cluster")

    plt.tight_layout()
    plt.show()