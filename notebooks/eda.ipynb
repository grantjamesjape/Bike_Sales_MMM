import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load data
df = pd.read_csv(r"data\bike_sales_data.csv")
df.head()

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
plt.savefig('charts\spend_correlation_matrix.png')
print("Saved: spend_correlation_matrix.png to showcase multicollinearity.")
plt.show()
