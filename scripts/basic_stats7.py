import pandas as pd
import numpy as np

# =========================================================
# LOAD DATA
# =========================================================

input_file = "/Users/bazam/dev/Italian_analysis/data/datacombined-italy2.csv"
df = pd.read_csv(input_file)

# =========================================================
# QUESTIONS
# =========================================================

questions = ["Q3", "Q9", "Q10", "Q19", "Q21", "Q29", "Q2", "Q4", "Q7", "Q20"]

# =========================================================
# FUNCTION: COUNT 0 / 1 / 2
# =========================================================

def count_response_distribution(df, questions, values=[0, 1, 2]):

    results = []

    for q in questions:

        counts = df[q].value_counts(dropna=False)

        row = {
            "Q": q
        }

        for v in values:
            row[v] = int(counts.get(v, 0))

        results.append(row)

    return pd.DataFrame(results).set_index("Q")

# =========================================================
# RUN
# =========================================================

count_table = count_response_distribution(df, questions)

# =========================================================
# OPTIONAL: ADD TOTAL COLUMN
# =========================================================

count_table["Total"] = count_table.sum(axis=1)

# =========================================================
# ROUND (not strictly needed but safe)
# =========================================================

count_table = count_table.astype(int)

# =========================================================
# EXPORT
# =========================================================

output_file = "question_response_counts.xlsx"

count_table.to_excel(output_file)

print("Done.")
print(f"Saved to: {output_file}")