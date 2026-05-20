import pandas as pd

# Load your dataset
# Replace with your actual file path
df = pd.read_csv("/Users/bazam/dev/Italian_analysis/data/all_answers_binary.csv")

# -----------------------------
# Define demographic columns
# -----------------------------

gender_cols = [
    "gender_female",
    "gender_male",
    "gender_non-binary",
    "gender_prefer_not_to_say"
]

age_cols = [
    "age_less_than_18",
    "age_19-30",
    "age_31-40",
    "age_41-55",
    "age_56-65",
    "age_over_65"
]

education_cols = [
    "educational_level_primary_school",
    "educational_level_high_school",
    "educational_level_bachelors_degree",
    "educational_level_masters_degree",
    "educational_level_PhD_or_equivalent"
]

# -----------------------------
# Helper function
# -----------------------------

def demographic_distribution(df_subset, columns, label_name):
    """
    Calculate counts and percentages for one-hot encoded demographic columns.
    """
    counts = df_subset[columns].sum().sort_values(ascending=False)
    percentages = (counts / len(df_subset) * 100).round(2)

    result = pd.DataFrame({
        label_name: counts.index,
        "count": counts.values,
        "percentage": percentages.values
    })

    return result

# =========================================================
# Q11_No == 1
# =========================================================

q11_group = df[df["Q11_No"] == 1]

print("\n==============================")
print("Q11_No == 1")
print("==============================")

print(f"\nNumber of respondents: {len(q11_group)}")

print("\nGender distribution")
print(demographic_distribution(q11_group, gender_cols, "gender"))

print("\nAge distribution")
print(demographic_distribution(q11_group, age_cols, "age_group"))

print("\nEducational level distribution")
print(demographic_distribution(q11_group, education_cols, "education_level"))

# =========================================================
# Q22_No == 1
# =========================================================

q22_group = df[df["Q22_No"] == 1]

print("\n==============================")
print("Q22_No == 1")
print("==============================")

print(f"\nNumber of respondents: {len(q22_group)}")

print("\nGender distribution")
print(demographic_distribution(q22_group, gender_cols, "gender"))

print("\nAge distribution")
print(demographic_distribution(q22_group, age_cols, "age_group"))

print("\nEducational level distribution")
print(demographic_distribution(q22_group, education_cols, "education_level"))