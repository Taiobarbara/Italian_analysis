import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("/Users/bazam/dev/Italian_analysis/data/datacombined-italy2.csv")

# Ensure Cluster is treated as categorical
df["Cluster"] = df["Cluster"].astype(str)

# Sort clusters numerically
clusters = sorted(df["Cluster"].unique(), key=int)

# Prepare data for violin plot
data = [df[df["Cluster"] == c]["knowledge_score"] for c in clusters]

plt.figure()
plt.violinplot(data, showmeans=True, showmedians=True)

plt.xticks(range(1, len(clusters) + 1), clusters)
plt.xlabel("Cluster")
plt.ylabel("Knowledge Score")
plt.title("Violin Plot of Knowledge Score per Cluster")

plt.show()