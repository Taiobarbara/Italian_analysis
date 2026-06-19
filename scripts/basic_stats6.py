import pandas as pd

# Load data
file_path = "/Users/bazam/dev/Italian_analysis/data/all_answers_binary.csv"
df = pd.read_csv(file_path)

# Demographic columns
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

edu_cols = [
    "educational_level_primary_school",
    "educational_level_high_school",
    "educational_level_bachelors_degree",
    "educational_level_masters_degree",
    "educational_level_PhD_or_equivalent"
]

demographic_groups = {
    "Gender": gender_cols,
    "Age": age_cols,
    "Education": edu_cols
}


def demographic_summary(df_subset, demographic_groups):
    """
    Calculate percentages for one-hot encoded demographic variables.
    """
    n = len(df_subset)

    results = []

    for category, cols in demographic_groups.items():
        for col in cols:
            pct = 100 * df_subset[col].sum() / n
            results.append({
                "Category": category,
                "Variable": col,
                "Count": int(df_subset[col].sum()),
                "Percent": round(pct, 1)
            })

    return pd.DataFrame(results)


def analyze_question(df, q_yes_col, q_no_col, question_name):
    print("\n" + "="*80)
    print(question_name)
    print("="*80)

    yes_group = df[df[q_yes_col] == 1]
    no_group = df[df[q_no_col] == 1]

    print(f"\nRespondents answering YES: {len(yes_group)}")
    yes_results = demographic_summary(yes_group, demographic_groups)

    print("\n--- YES group ---")
    print(yes_results)

    print(f"\nRespondents answering NO: {len(no_group)}")
    no_results = demographic_summary(no_group, demographic_groups)

    print("\n--- NO group ---")
    print(no_results)

    return yes_results, no_results


# Q11
q11_yes, q11_no = analyze_question(
    df,
    q_yes_col="Q11_Yes",
    q_no_col="Q11_No",
    question_name="Q11"
)

# Q22
q22_yes, q22_no = analyze_question(
    df,
    q_yes_col="Q22_Yes",
    q_no_col="Q22_No",
    question_name="Q22"
)
with pd.ExcelWriter("demographic_summary_Q11_Q22.xlsx") as writer:
    q11_yes.to_excel(writer, sheet_name="Q11_Yes", index=False)
    q11_no.to_excel(writer, sheet_name="Q11_No", index=False)
    q22_yes.to_excel(writer, sheet_name="Q22_Yes", index=False)
    q22_no.to_excel(writer, sheet_name="Q22_No", index=False)

print("\nResults saved to demographic_summary_Q11_Q22.xlsx")