import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, kruskal

# =========================================================
# LOAD DATA
# =========================================================
input_file = "/Users/bazam/dev/Italian_analysis/data/datacombined-italy2.csv"
df = pd.read_csv(input_file)

# =========================================================
# DEFINE GROUPS
# =========================================================
gender_cols = [
    "gender_female",
    "gender_male"
]

education_cols = [
    "educational_level_high_school",
    "educational_level_bachelors_degree",
    "educational_level_masters_degree",
    "educational_level_PhD_or_equivalent"
]

age_cols = [
    "age_19-30",
    "age_31-40",
    "age_41-55",
    "age_56-65",
    "age_over_65"
]

score_col = "knowledge_score"

# =========================================================
# SCORE INTERVALS
# =========================================================
bins = [0, 0.25, 0.5, 0.75, 1.000001]
labels = [
    "0-0.25",
    "0.25-0.5",
    "0.5-0.75",
    "0.75-1"
]

# =========================================================
# FUNCTION: INTERVAL DISTRIBUTION
# =========================================================
def calculate_interval_distribution(df, group_cols, score_col):
    results = []
    for group in group_cols:
        subset = df[df[group] == 1]
        total_n = len(subset)
        categorized = pd.cut(
            subset[score_col],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        percentages = (
            categorized.value_counts(normalize=True)
            .reindex(labels, fill_value=0)
            * 100
        )
        row = {"Group": group}
        for label in labels:
            row[label] = percentages[label]
        results.append(row)
    return pd.DataFrame(results)

# =========================================================
# FUNCTION: DESCRIPTIVE STATISTICS
# =========================================================
def calculate_descriptive_stats(df, group_cols, score_col):
    results = []
    for group in group_cols:
        subset = df[df[group] == 1][score_col]
        row = {
            "Group": group,
            "Mean": subset.mean(),
            "Median": subset.median(),
            "SD": subset.std()
        }
        results.append(row)
    return pd.DataFrame(results)

# =========================================================
# INTERVAL DISTRIBUTIONS
# =========================================================
gender_intervals = calculate_interval_distribution(df, gender_cols, score_col)
education_intervals = calculate_interval_distribution(df, education_cols, score_col)
age_intervals = calculate_interval_distribution(df, age_cols, score_col)

# =========================================================
# DESCRIPTIVE STATISTICS
# =========================================================
gender_stats = calculate_descriptive_stats(df, gender_cols, score_col)
education_stats = calculate_descriptive_stats(df, education_cols, score_col)
age_stats = calculate_descriptive_stats(df, age_cols, score_col)

# =========================================================
# STATISTICAL TESTS
# =========================================================

# -------------------------
# Mann-Whitney U (Gender)
# -------------------------
female_scores = df[df["gender_female"] == 1][score_col]
male_scores = df[df["gender_male"] == 1][score_col]
u_stat, u_p = mannwhitneyu(female_scores, male_scores, alternative='two-sided')
gender_test = pd.DataFrame({
    "Test": ["Mann-Whitney U"],
    "Statistic": [u_stat],
    "p_value": [u_p]
})

# -------------------------
# Kruskal-Wallis (Education)
# -------------------------
education_groups = [df[df[col] == 1][score_col] for col in education_cols]
h_edu, p_edu = kruskal(*education_groups)
education_test = pd.DataFrame({
    "Test": ["Kruskal-Wallis"],
    "Statistic": [h_edu],
    "p_value": [p_edu]
})

# -------------------------
# Kruskal-Wallis (Age)
# -------------------------
age_groups = [df[df[col] == 1][score_col] for col in age_cols]
h_age, p_age = kruskal(*age_groups)
age_test = pd.DataFrame({
    "Test": ["Kruskal-Wallis"],
    "Statistic": [h_age],
    "p_value": [p_age]
})

# -------------------------
# Kruskal-Wallis (Cluster vs Knowledge Score)
# -------------------------

# Get unique cluster IDs
cluster_ids = sorted(df["Cluster"].dropna().unique())

# Create one score vector per cluster
cluster_groups = [
    df[df["Cluster"] == cluster][score_col]
    for cluster in cluster_ids
]

# Run Kruskal-Wallis test
h_cluster, p_cluster = kruskal(*cluster_groups)

cluster_test = pd.DataFrame({
    "Test": ["Kruskal-Wallis"],
    "Statistic": [h_cluster],
    "p_value": [p_cluster]
})

print(cluster_test)

# =========================================================
# KRUSKAL-WALLIS FOR KNOWLEDGE QUESTIONS BY CLUSTER
# =========================================================

knowledge_questions = [
    "Q1",
    "Q5",
    "Q8",
    "Q11",
    "Q12",
    "Q13",
    "Q14",
    "Q16",
    "Q18",
    "Q22",
    "Q26",
    "Q27"
]

question_results = []

# Get cluster IDs
cluster_ids = sorted(df["Cluster"].dropna().unique())

for question in knowledge_questions:

    # Create one group per cluster
    groups = [
        df[df["Cluster"] == cluster][question].dropna()
        for cluster in cluster_ids
    ]

    # Run Kruskal-Wallis
    h_stat, p_value = kruskal(*groups)

    # Store results
    question_results.append({
        "Question": question,
        "Statistic": h_stat,
        "p_value": p_value
    })

# Convert to dataframe
knowledge_question_tests = pd.DataFrame(question_results)

# Round values
numeric_cols = knowledge_question_tests.select_dtypes(include=np.number).columns
knowledge_question_tests[numeric_cols] = (
    knowledge_question_tests[numeric_cols].round(4)
)

print(knowledge_question_tests)

# =========================================================
# ROUND RESULTS
# =========================================================
tables_to_round = [
    gender_intervals, education_intervals, age_intervals,
    gender_stats, education_stats, age_stats,
    gender_test, education_test, age_test, cluster_test
]

for table in tables_to_round:
    numeric_cols = table.select_dtypes(include=np.number).columns
    table[numeric_cols] = table[numeric_cols].round(4)

# =========================================================
# EXPORT TO EXCEL
# =========================================================
output_file = "knowledge_score_cluster_analysis.xlsx"

with pd.ExcelWriter(output_file) as writer:
    gender_intervals.to_excel(writer, sheet_name="Gender_Intervals", index=False)
    education_intervals.to_excel(writer, sheet_name="Education_Intervals", index=False)
    age_intervals.to_excel(writer, sheet_name="Age_Intervals", index=False)
    gender_stats.to_excel(writer, sheet_name="Gender_Stats", index=False)
    education_stats.to_excel(writer, sheet_name="Education_Stats", index=False)
    age_stats.to_excel(writer, sheet_name="Age_Stats", index=False)
    gender_test.to_excel(writer, sheet_name="Gender_Test", index=False)
    education_test.to_excel(writer, sheet_name="Education_Test", index=False)
    age_test.to_excel(writer, sheet_name="Age_Test", index=False)
    cluster_test.to_excel(writer, sheet_name="Cluster_Test", index=False)

knowledge_question_tests.to_excel(
    writer,
    sheet_name="Knowledge_Q_Cluster_Test",
    index=False
)

print("Analysis complete.")
print(f"Results saved to: {output_file}")