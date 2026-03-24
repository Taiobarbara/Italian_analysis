import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load dataset
df = pd.read_csv("/Users/bazam/dev/Italian_analysis/data/datacombined-italy2.csv")
df["Cluster"] = df["Cluster"].astype(str)

# Sort clusters numerically
clusters = sorted(df["Cluster"].unique(), key=int)

plt.figure()

y_offsets = np.arange(len(clusters))  # vertical spacing

for i, cluster in enumerate(clusters):
    cluster_data = df[df["Cluster"] == cluster]["knowledge_score"].dropna()
    
    # Create KDE
    kde = gaussian_kde(cluster_data)
    
    # Define x range based on knowledge score scale
    x_vals = np.linspace(df["knowledge_score"].min(),
                         df["knowledge_score"].max(), 500)
    
    density = kde(x_vals)
    
    # Normalize density height for cleaner stacking
    density = density / density.max() * 0.8
    
    # Plot filled density shifted vertically
    plt.fill_between(x_vals, y_offsets[i],
                     y_offsets[i] + density)
    
    # Optional outline
    plt.plot(x_vals, y_offsets[i] + density)

plt.yticks(y_offsets + 0.4, clusters)
plt.xlabel("Knowledge Score")
plt.ylabel("Cluster")
plt.title("Ridge Line Plot of Knowledge Score per Cluster")

plt.show()