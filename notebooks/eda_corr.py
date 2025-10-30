import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# define standard folder names
DATA_DIR = 'data'
CHARTS_DIR = 'charts'

# ensure the output directory exists before saving (critical for running scripts)
os.makedirs(CHARTS_DIR, exist_ok=True)

# load data
data_path = os.path.join(DATA_DIR, 'bike_sales_data.csv')
df = pd.read_csv(data_path) 
print("Data loaded successfully.")
print(df.head())

# identify marketing spend columns
spend_cols = [col for col in df.columns if 'spend' in col]
print("Marketing Spend columns:", spend_cols)

# check for multicollinearity (correlation matrix)
plt.figure(figsize=(10, 8))
correlation_matrix = df[spend_cols].corr()
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt=".2f",
    linewidths=.5,
    cbar_kws={'label': 'Correlation Coefficient'}
)
plt.title('Correlation Matrix of Marketing Spend Channels (Multicollinearity Check)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

output_path = os.path.join(CHARTS_DIR, 'spend_correlation_matrix.png')
plt.savefig(output_path)
