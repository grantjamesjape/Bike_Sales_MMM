import pandas as pd
import statsmodels.formula.api as smf
import os

# Define standard folder names based on your project structure
REPORTS_DIR = 'report' 
# Assuming bike_sales_data_transformed.csv was saved to the root directory
TRANSFORMED_DATA_PATH = 'bike_sales_data_transformed.csv' 

# Ensure the output directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)


# --- Load the transformed data (Path Corrected for Portability) ---
transformed_df = pd.read_csv(TRANSFORMED_DATA_PATH) 

# --- Fix the Perfect Collinearity (r=1.00) ---
branded_fb_col = 'branded_search_spend_adstock_sat'
facebook_col = 'facebook_spend_adstock_sat'
transformed_df['search_social_spend_adstock_sat'] = transformed_df[branded_fb_col] + transformed_df[facebook_col]
transformed_df = transformed_df.drop(columns=[branded_fb_col, facebook_col])

# --- Define and Fit the Advanced Model ---
new_spend_cols = [col for col in transformed_df.columns if '_adstock_sat' in col]
formula = "sales ~ " + " + ".join(new_spend_cols)

print("Final Model Formula:")
print(formula)

model_advanced = smf.ols(formula=formula, data=transformed_df).fit()

print("\nAdvanced OLS Regression Results (Adstock & Saturation with Combined Channel):")
print(model_advanced.summary().as_text())

# --- Calculate Transformed Marginal ROI/Sensitivity ---
roi_df_advanced = model_advanced.params.to_frame(name='Coefficient').reset_index().rename(columns={'index': 'Channel'})
pvalues_df_advanced = model_advanced.pvalues.to_frame(name='P_Value').reset_index().rename(columns={'index': 'Channel'})

final_results = pd.merge(roi_df_advanced, pvalues_df_advanced, on='Channel')
final_results = final_results[final_results['Channel'] != 'Intercept'].copy()

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

final_results['P_Value'] = final_results['P_Value'].round(4)
final_results['Coefficient'] = final_results['Coefficient'].round(2)
final_results['is_significant'] = final_results['P_Value'] < 0.05
final_results['significance_label'] = final_results['is_significant'].apply(lambda x: 'Significant (P<0.05)' if x else 'Not Significant (P>=0.05)')

final_results = final_results.sort_values(by='Coefficient', ascending=False)


# --- Final Output Table and Saving (Path Corrected) ---
final_output_table = final_results[['Channel', 'Coefficient', 'P_Value', 'significance_label']]

print("\nFinal Advanced MMM Model Results:")
print(final_output_table.to_string(index=False))

# Use os.path.join to save the final results into the 'reports' folder
final_output_path = os.path.join(REPORTS_DIR, 'final_mmm_results.csv')
final_output_table.to_csv(final_output_path, index=False)

print(f"\nSaved final results to: {final_output_path} for your portfolio.")
