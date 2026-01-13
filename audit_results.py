import pandas as pd
import ast
df = pd.read_csv('results/results_both_combined.csv', nrows=1)
l = ast.literal_eval(df['cpu_times_per_step'].iloc[0])
print(f"List length: {len(l)}")
print(f"First 10 values: {l[:10]}")
