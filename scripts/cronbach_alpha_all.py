import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("/Users/bazam/dev/Italian_analysis/data/datacombined-italy2.csv")

# ---------- Function to compute Cronbach's Alpha ----------
def cronbach_alpha(items_df):
    items_df = items_df.dropna()
    item_scores = items_df.to_numpy()
    
    item_variances = item_scores.var(axis=0, ddof=1)
    total_score = item_scores.sum(axis=1)
    total_variance = total_score.var(ddof=1)
    
    n_items = items_df.shape[1]
    
    alpha = (n_items / (n_items - 1)) * \
            (1 - (item_variances.sum() / total_variance))
    
    return alpha

# ---------- Define scale items ----------
knowledge_items = ["Q1","Q5","Q8","Q11","Q12","Q13","Q14","Q16","Q18","Q22","Q26","Q27"]
attitude_items = ["Q3","Q9","Q10","Q19","Q21","Q29"] 
practice_items = ["Q2","Q4","Q20", "Q7"]
risk_items = ["Q6","Q15","Q17","Q23","Q24","Q25"]

# ---------- Compute alphas ----------
alpha_knowledge = cronbach_alpha(df[knowledge_items])
alpha_attitude = cronbach_alpha(df[attitude_items])
alpha_practice = cronbach_alpha(df[practice_items])
alpha_risk = cronbach_alpha(df[risk_items])

# ---------- Subsets ----------
df_female = df[df["gender_female"] == 1]
df_male = df[df["gender_male"] == 1]

# ---------- Compute alphas ----------
def compute_all_alphas(data, label):
    print(f"\nCronbach's Alpha - {label}")
    print(f"Knowledge: {cronbach_alpha(data[knowledge_items]):.4f}")
    print(f"Attitude:  {cronbach_alpha(data[attitude_items]):.4f}")
    print(f"Practice:  {cronbach_alpha(data[practice_items]):.4f}")
    print(f"Risk:      {cronbach_alpha(data[risk_items]):.4f}")

# Full sample
compute_all_alphas(df, "Full Sample")

# Female
compute_all_alphas(df_female, "Female")

# Male
compute_all_alphas(df_male, "Male")