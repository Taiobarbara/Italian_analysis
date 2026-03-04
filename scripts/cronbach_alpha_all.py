import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("/Users/bazam/dev/Italian_analysis/data/datacombined-italy.csv")

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
attitude_items = ["Q10","Q14","Q19","Q21","Q24","Q29"] #"Q8","Q9",
knowledge_items = ["Q1","Q5","Q11","Q12","Q13","Q16","Q18","Q22","Q23","Q25","Q27"]
practice_items = ["Q2","Q3","Q4","Q20"] # "Q7","Q30"
risk_items = ["Q6","Q15","Q17","Q26"]

# ---------- Compute alphas ----------
alpha_attitude = cronbach_alpha(df[attitude_items])
alpha_knowledge = cronbach_alpha(df[knowledge_items])
alpha_practice = cronbach_alpha(df[practice_items])
alpha_risk = cronbach_alpha(df[risk_items])

# ---------- Print results ----------
print("Cronbach's Alpha Results:")
print(f"Attitude scale:  {alpha_attitude:.4f}")
print(f"Knowledge scale: {alpha_knowledge:.4f}")
print(f"Practice scale:  {alpha_practice:.4f}")
print(f"Risk scale:      {alpha_risk:.4f}")


# ---------- Exclude Cluster 0 from risk ----------
df_no_cluster0 = df[df["Cluster"] != 0]

# ---------- Compute alpha ----------
alpha_risk_no_cluster0 = cronbach_alpha(df_no_cluster0[risk_items])

print("Cronbach's Alpha for Risk Scale (excluding Cluster 0):")
print(f"{alpha_risk_no_cluster0:.4f}")