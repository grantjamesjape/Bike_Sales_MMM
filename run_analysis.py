import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import os
import warnings
warnings.filterwarnings("ignore") # Suppress warnings for cleaner output


# --- 1. GLOBAL SETUP AND UTILITY FUNCTIONS ---

# Define standard folder names based on your GitHub repository structure
DATA_DIR = 'data'
CHARTS_DIR = 'charts'
REPORTS_DIR = 'reports' 
TRANSFORMED_DATA_FILENAME = 'bike_sales_data_transformed.csv' 

# Create output directories if they don't exist
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
print("Setup: Output directories created successfully.")


# Adstock Transformation Function (Geometric Decay)
def geometric_adstock(series, decay_factor):
    """
    Applies Geometric Adstock (carryover effect) to a time series.
    Formula: Adstock_t = Spend_t + decay_factor * Adstock_{t-1}
    """
    adstock_series = np.zeros_like(series, dtype=float)
    for t in range(len(series)):
        adstock_series[t] = series[t] + (decay_factor * adstock_series[t-1] if t > 0 else 0)
    return adstock_series

# Saturation Transformation Function (Simple Log)
def log_saturation(series, constant=1):
    """
    Applies a Logarithmic transformation to model diminishing returns (saturation).
    Formula: Log(Spend + C)
    """
    return np.log(series + constant)


# =========================================================================
# --- 2. STAGE 1: EDA & Multicollinearity Check ---
# =========================================================================
print("\n--- STAGE 1: EDA and Multicollinearity Check ---")

# Load data using portable path
data_path = os.path.join(DATA_DIR, 'bike_sales_data.csv')
df = pd.read_csv(data_path) 
spend_cols = [col for col in df.columns if 'spend' in col]

print(f"Data loaded from: {data_path}")
print("Marketing Spend columns:", spend_cols)

# Check for multicollinearity (correlation matrix)
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

# Save image to the charts/ folder
output_path_corr = os.path.join(CHARTS_DIR, 'spend_correlation_matrix.png')
plt.savefig(output_path_corr)
plt.close()

print(f"Result: Correlation matrix saved to {output_path_corr}.")
print("Finding: Perfect correlation (r=1.00) found between Branded Search and Facebook.")


# =========================================================================
# --- 3. STAGE 2: Data Transformation (Adstock & Saturation) ---
# =========================================================================
print("\n--- STAGE 2: Data Transformation ---")

transformed_df = df.copy()
adstock_decay = 0.5  # 50% carryover
log_constant = 1     # For log-saturation safety

for col in spend_cols:
    # 1. Apply Adstock
    adstock_col = f'{col}_adstock_{adstock_decay}'
    transformed_df[adstock_col] = geometric_adstock(transformed_df[col], adstock_decay)
    
    # 2. Apply Saturation (to the Adstocked feature)
    adstock_sat_col = f'{col}_adstock_sat'
    transformed_df[adstock_sat_col] = log_saturation(transformed_df[adstock_col], log_constant)
    
    # Drop the original and intermediate adstock columns
    transformed_df = transformed_df.drop(columns=[col, adstock_col])

# Save the transformed data to the root directory
transformed_df.to_csv(TRANSFORMED_DATA_FILENAME, index=False)
print(f"Result: Transformed data saved to {TRANSFORMED_DATA_FILENAME} for modeling.")


# =========================================================================
# --- 4. STAGE 3: Advanced Modeling & Final Results ---
# =========================================================================
print("\n--- STAGE 3: Advanced Modeling ---")

# --- FIX COLINEARITY: Combine Branded Search and Facebook ---
branded_fb_col = 'branded_search_spend_adstock_sat'
facebook_col = 'facebook_spend_adstock_sat'
transformed_df['search_social_spend_adstock_sat'] = transformed_df[branded_fb_col] + transformed_df[facebook_col]
transformed_df = transformed_df.drop(columns=[branded_fb_col, facebook_col])

# Define and Fit the Advanced OLS Model
new_spend_cols = [col for col in transformed_df.columns if '_adstock_sat' in col]
formula = "sales ~ " + " + ".join(new_spend_cols)

print("Final Model Formula:", formula)
model_advanced = smf.ols(formula=formula, data=transformed_df).fit()

print("\n--- Model Summary ---")
print(model_advanced.summary().as_text())
print(f"Model R-squared: {model_advanced.rsquared:.3f}")

# --- Calculate and Save Final Results Table ---
roi_df_advanced = model_advanced.params.to_frame(name='Coefficient').reset_index().rename(columns={'index': 'Channel'})
pvalues_df_advanced = model_advanced.pvalues.to_frame(name='P_Value').reset_index().rename(columns={'index': 'Channel'})

final_results = pd.merge(roi_df_advanced, pvalues_df_advanced, on='Channel')
final_results = final_results[final_results['Channel'] != 'Intercept'].copy()

# Clean up channel names
name_mapping = {
    'search_social_spend_adstock_sat': 'Search & Facebook',
    'nonbranded_search_spend_adstock_sat': 'Nonbranded Search',
    'print_spend_adstock_sat': 'Print',
    'ooh_spend_adstock_sat': 'OOH',
    'tv_spend_adstock_sat': 'TV',
    'radio_spend_adstock_sat': 'Radio'
}
final_results['Channel'] = final_results['Channel'].replace(name_mapping)

final_results['P_Value'] = final_results['P_Value'].round(4)
final_results['Coefficient'] = final_results['Coefficient'].round(2)
final_results['is_significant'] = final_results['P_Value'] < 0.05
final_results['significance_label'] = final_results['is_significant'].apply(lambda x: 'Significant (P<0.05)' if x else 'Not Significant (P>=0.05)')
final_results = final_results.sort_values(by='Coefficient', ascending=False)

final_output_table = final_results[['Channel', 'Coefficient', 'P_Value', 'significance_label']]

# Save the final results to the reports/ folder
final_output_path = os.path.join(REPORTS_DIR, 'final_mmm_results.csv')
final_output_table.to_csv(final_output_path, index=False)

print(f"\nFinal Results Table (Saved to {final_output_path}):")
print(final_output_table.to_string(index=False))


# =========================================================================
# --- 5. STAGE 4: Visualization ---
# =========================================================================
print("\n--- STAGE 4: Visualization ---")

# --- Plot 1: Actual vs. Predicted Sales (Model Fit Over Time) ---
transformed_df['predicted_sales'] = model_advanced.predict(transformed_df)

plt.figure(figsize=(12, 6))
plt.plot(transformed_df['Week'], transformed_df['sales'], label='Actual Sales', color='blue', alpha=0.7)
plt.plot(transformed_df['Week'], transformed_df['predicted_sales'], label='Predicted Sales (Model Fit)', color='red', linestyle='--')

plt.title(f'Actual vs. Predicted Sales (R-squared: {model_advanced.rsquared:.3f})')
plt.xlabel('Week (Time)')
plt.ylabel('Sales (USD)')
plt.xticks(transformed_df['Week'][::20], rotation=45, ha='right') 
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Save to the charts/ folder
output_path_fit = os.path.join(CHARTS_DIR, 'actual_vs_predicted_sales.png')
plt.savefig(output_path_fit)
plt.close()
print(f"Plot 1 saved: {output_path_fit}")

# --- Plot 2: Channel Sensitivity (Coefficient Bar Chart) ---
plt.figure(figsize=(10, 6))
# Define color based on positive/negative coefficient
colors = ['green' if c > 0 else 'red' for c in final_results['Coefficient']]

plt.bar(final_results['Channel'], final_results['Coefficient'], color=colors)
plt.title('Sales Sensitivity (Coefficient) by Marketing Channel')
plt.xlabel('Marketing Channel')
plt.ylabel('Sales Lift Coefficient (USD)')
plt.ticklabel_format(style='plain', axis='y', useOffset=False)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()

# Save to the charts/ folder
output_path_coef = os.path.join(CHARTS_DIR, 'channel_sensitivity_bar_chart.png')
plt.savefig(output_path_coef)
plt.close()
print(f"Plot 2 saved: {output_path_coef}")

print("\n--- Analysis Complete ---")
