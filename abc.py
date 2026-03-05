import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
df = pd.read_csv(url)

df_region3 = df[df['Region'] == 3]

group1 = df_region3[df_region3['Channel'] == 1]['Detergents_Paper']
group2 = df_region3[df_region3['Channel'] == 2]['Detergents_Paper']

n1 = len(group1)
n2 = len(group2)

x_bar1 = group1.mean()
x_bar2 = group2.mean()

s1 = group1.std()
s2 = group2.std()

print("--- Step 1: Group Statistics ---")
print(f"Group 1 (Horeca) Sample Size (n1): {n1}")
print(f"Group 2 (Retail) Sample Size (n2): {n2}")
print(f"Group 1 Mean (x_bar1): {x_bar1:.4f}")
print(f"Group 2 Mean (x_bar2): {x_bar2:.4f}")
print(f"Group 1 Standard Deviation (s1): {s1:.4f}")
print(f"Group 2 Standard Deviation (s2): {s2:.4f}\n")

numerator = (n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)
denominator = n1 + n2 - 2
sp_squared = numerator / denominator
sp = np.sqrt(sp_squared)

print("--- Step 2: Pooled Standard Deviation ---")
print(f"Numerator [(n1-1)*s1^2 + (n2-1)*s2^2]: {numerator:.4f}")
print(f"Denominator [n1 + n2 - 2]: {denominator}")
print(f"Pooled Variance (sp^2): {sp_squared:.4f}")
print(f"Pooled Standard Deviation (sp): {sp:.4f}\n")

mean_difference = x_bar2 - x_bar1
d = mean_difference / sp

print("--- Step 3: Cohen's d ---")
print(f"Mean Difference (x_bar2 - x_bar1): {mean_difference:.4f}")
print(f"Formula: {mean_difference:.4f} / {sp:.4f}")
print(f"Final Cohen's d: {d:.3f}")

print("\n--- Final Requested Output ---")
print(f"{d:.3f}")
