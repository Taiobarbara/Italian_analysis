import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------

file_path = "/Users/bazam/dev/Italian_analysis/data/all_answers_binary.csv"
df = pd.read_csv(file_path)

# ------------------------------------------------------------------
# Convert one-hot encoded demographics into categorical variables
# ------------------------------------------------------------------

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


def onehot_to_category(df, columns, new_name):
    df[new_name] = (
        df[columns]
        .idxmax(axis=1)
        .str.replace(new_name + "_", "", regex=False)
        .str.replace("educational_level_", "", regex=False)
        .str.replace("gender_", "", regex=False)
        .str.replace("age_", "", regex=False)
    )


onehot_to_category(df, gender_cols, "gender")
onehot_to_category(df, age_cols, "age")
onehot_to_category(df, education_cols, "educational_level")

# ------------------------------------------------------------------
# Convert Yes/No questions into categorical variables
# ------------------------------------------------------------------

df["Q11"] = np.where(df["Q11_Yes"] == 1, "Yes", "No")
df["Q22"] = np.where(df["Q22_Yes"] == 1, "Yes", "No")


# ------------------------------------------------------------------
# Chi-square function
# ------------------------------------------------------------------

def cramers_v(contingency):
    chi2 = chi2_contingency(contingency)[0]
    n = contingency.values.sum()

    r, k = contingency.shape

    return np.sqrt(chi2 / (n * (min(r - 1, k - 1))))


def run_chi_square(df, question, demographic):

    print("=" * 80)
    print(f"{question} vs {demographic}")
    print("=" * 80)

    contingency = pd.crosstab(df[demographic], df[question])

    print("\nObserved counts")
    print(contingency)

    chi2, p, dof, expected = chi2_contingency(contingency)

    print("\nExpected counts")
    print(
        pd.DataFrame(
            expected,
            index=contingency.index,
            columns=contingency.columns
        ).round(2)
    )

    print(f"\nChi-square = {chi2:.3f}")
    print(f"Degrees of freedom = {dof}")
    print(f"P-value = {p:.5f}")
    print(f"Cramer's V = {cramers_v(contingency):.3f}")

    if p < 0.05:
        print("Result: Significant association")
    else:
        print("Result: No significant association")

    print("\n")


# ------------------------------------------------------------------
# Run analyses
# ------------------------------------------------------------------

results = []

for question in ["Q11", "Q22"]:
    for demographic in ["gender", "age", "educational_level"]:

        contingency = pd.crosstab(df[demographic], df[question])

        chi2, p, dof, expected = chi2_contingency(contingency)

        results.append({
            "Question": question,
            "Demographic": demographic,
            "Chi-square": chi2,
            "df": dof,
            "p-value": p,
            "Cramers_V": cramers_v(contingency)
        })

results = pd.DataFrame(results)

print(results)

results.to_csv("chi_square_summary.csv", index=False)

pd.crosstab(
    df["age"],
    df["Q11"],
    normalize="index"
).round(3) * 100

pd.crosstab(
    df["educational_level"],
    df["Q11"],
    normalize="index"
).round(3) * 100