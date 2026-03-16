import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("/Users/bazam/dev/Italian_analysis/data/datacombined-italy2.csv")

def mcdonalds_omega(items_df):
    
    # Clean data
    items_df = items_df.dropna().astype(float)
    
    # Remove zero-variance items
    items_df = items_df.loc[:, items_df.var() > 0]
    
    # Covariance matrix
    cov_matrix = np.cov(items_df, rowvar=False)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Get first (largest) eigenvalue/vector
    idx = np.argmax(eigenvalues)
    first_eigenvalue = eigenvalues[idx]
    first_eigenvector = eigenvectors[:, idx]
    
    # Compute loadings
    loadings = np.sqrt(first_eigenvalue) * first_eigenvector
    
    # Unique variances
    communalities = loadings**2
    total_variances = np.diag(cov_matrix)
    unique_variances = total_variances - communalities
    
    # Omega total
    numerator = (np.sum(loadings))**2
    denominator = numerator + np.sum(unique_variances)
    
    omega = numerator / denominator
    
    return float(np.real(omega))

knowledge_items = ["Q1","Q5","Q8","Q11","Q12","Q13","Q14","Q16","Q18","Q22","Q26","Q27"]
attitude_items = ["Q3","Q9","Q10","Q19","Q21","Q29"] 
practice_items = ["Q2","Q4","Q20", "Q7"]
risk_items = ["Q6","Q15","Q17","Q23","Q24","Q25"]

print("McDonald's Omega")
print(f"Knowledge: {mcdonalds_omega(df[knowledge_items]):.4f}")
print(f"Attitude:  {mcdonalds_omega(df[attitude_items]):.4f}")
print(f"Practice:  {mcdonalds_omega(df[practice_items]):.4f}")
print(f"Risk:      {mcdonalds_omega(df[risk_items]):.4f}")