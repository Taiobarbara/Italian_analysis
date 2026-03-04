import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load dataset
df = pd.read_csv("/Users/bazam/dev/Italian_analysis/data/datacombined-italy.csv")
df["Cluster"] = df["Cluster"].astype(str)

# Sort clusters numerically
clusters = sorted(df["Cluster"].unique(), key=int)

plt.figure()

y_offsets = np.arange(len(clusters))  # vertical spacing

for i, cluster in enumerate(clusters):
    cluster_data = df[df["Cluster"] == cluster]["practice_score"].dropna()
    
    # Create KDE
    kde = gaussian_kde(cluster_data)
    
    # Define x range based on attitude score scale
    x_vals = np.linspace(df["practice_score"].min(),
                         df["practice_score"].max(), 500)
    
    density = kde(x_vals)
    
    # Normalize density height for cleaner stacking
    density = density / density.max() * 0.8
    
    # Plot filled density shifted vertically
    plt.fill_between(x_vals, y_offsets[i],
                     y_offsets[i] + density)
    
    # Optional outline
    plt.plot(x_vals, y_offsets[i] + density)

plt.yticks(y_offsets + 0.4, clusters)
plt.xlabel("Practice Score")
plt.ylabel("Cluster")
plt.title("Ridge Line Plot of Practice Score per Cluster")

plt.show()