import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import os # Import os for robust file path handling

# --- Define standard folder names based on your project structure ---
CHARTS_DIR = 'charts' 
TRANSFORMED_DATA_PATH = 'bike_sales_data_transformed.csv' 

# Ensure the output directory exists
os.makedirs(CHARTS_DIR, exist_ok=True)


# --- Setup: Load data and Model Definition (Prerequisite for Plotting) ---
# Load the transformed data from the root directory
transformed_df = pd.read_csv(TRANSFORMED_DATA_PATH) 

# Fix the Perfect Collinearity (r=1.00) by combining the features
branded_fb_col = 'branded_search_spend_adstock_sat'
facebook_col = 'facebook_spend_adstock_sat'
transformed_df['search_social_spend_adstock_sat'] = transformed_df[branded_fb_col] + transformed_df[facebook_col]
transformed_df = transformed_df.drop(columns=[branded_fb_col, facebook_col])

# Define and Fit the Advanced OLS Model
new_spend_cols = [col for col in transformed_df.columns if '_adstock_sat' in col]
formula = "sales ~ " + " + ".join(new_spend_cols)
model_advanced = smf.ols(formula=formula, data=transformed_df).fit()


# =========================================================================
# --- Plot 1: Actual vs. Predicted Sales (Model Fit Over Time) ---
# =========================================================================

# Generate predictions from the model
transformed_df['predicted_sales'] = model_advanced.predict(transformed_df)

plt.figure(figsize=(12, 6))
plt.plot(transformed_df['Week'], transformed_df['sales'], 
         label='Actual Sales', 
         color='blue', 
         alpha=0.7)
plt.plot(transformed_df['Week'], transformed_df['predicted_sales'], 
         label='Predicted Sales (Model Fit)', 
         color='red', 
         linestyle='--')

plt.title(f'Actual vs. Predicted Sales (R-squared: {model_advanced.rsquared:.3f})')
plt.xlabel('Week (Time)')
plt.ylabel('Sales (USD)')

# Adjust x-axis to show labels clearly for a long time series
plt.xticks(transformed_df['Week'][::20], rotation=45, ha='right') 
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# CORRECTED SAVE PATH: Saves to the charts/ folder
output_path_1 = os.path.join(CHARTS_DIR, 'actual_vs_predicted_sales.png')
plt.savefig(output_path_1)

plt.close()


# =========================================================================
# --- Plot 2: Channel Sensitivity (Coefficient Bar Chart) ---
# =========================================================================

# Extract and clean coefficients for plotting
final_results = model_advanced.params.to_frame(name='Coefficient').reset_index().rename(columns={'index': 'Channel'})
final_results = final_results[final_results['Channel'] != 'Intercept'].copy() # Exclude the Intercept

# Clean up channel names for the final report
name_mapping = {
    'search_social_spend_adstock_sat': 'Search & Facebook',
    'nonbranded_search_spend_adstock_sat': 'Nonbranded Search',
    'print_spend_adstock_sat': 'Print',
    'ooh_spend_adstock_sat': 'OOH',
    'tv_spend_adstock_sat': 'TV',
    'radio_spend_adstock_sat': 'Radio'
}
final_results['Channel'] = final_results['Channel'].replace(name_mapping)

# Sort the channels by coefficient value (descending)
final_results = final_results.sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
# Define color based on positive/negative coefficient
colors = ['green' if c > 0 else 'red' for c in final_results['Coefficient']]

plt.bar(final_results['Channel'], final_results['Coefficient'], color=colors)
plt.title('Sales Sensitivity (Coefficient) by Marketing Channel')
plt.xlabel('Marketing Channel')
plt.ylabel('Sales Lift Coefficient (USD)')

# Format y-axis to show currency style
plt.ticklabel_format(style='plain', axis='y', useOffset=False)

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()

# CORRECTED SAVE PATH: Saves to the charts/ folder
output_path_2 = os.path.join(CHARTS_DIR, 'channel_sensitivity_bar_chart.png')
plt.savefig(output_path_2)

plt.close()

print(f"Graphs successfully saved to the '{CHARTS_DIR}' folder.")
print(f"Files saved: '{output_path_1}' and '{output_path_2}'.")
