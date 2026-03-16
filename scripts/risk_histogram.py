import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/Users/bazam/dev/Italian_analysis/data/datacombined-italy2.csv")

risk_items = ["Q6","Q15","Q17","Q23","Q24","Q25"]

sns.set(style="whitegrid")

for q in risk_items:

    # Compute proportions per cluster
    prop_df = (
        df.groupby("Cluster")[q]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )

    # Convert to percentage
    prop_df["percentage"] = prop_df["proportion"] * 100

    # Plot
    g = sns.FacetGrid(
        prop_df,
        col="Cluster",
        col_wrap=4,
        height=4,
        sharex=True,
        sharey=True
    )

    g.map_dataframe(
        sns.barplot,
        x=q,
        y="percentage",
        order=sorted(df[q].dropna().unique())
    )

    g.set_axis_labels(q + " response", "Percentage of respondents")
    g.set_titles("Cluster {col_name}")

    plt.suptitle(f"Percentage Distribution of {q} by Cluster", y=1.05)

    plt.tight_layout()
    plt.savefig(f"risk_percentage_{q}_by_cluster.png", dpi=300)
    plt.show()