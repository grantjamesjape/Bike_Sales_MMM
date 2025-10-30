import pandas as pd
import numpy as np
import os

# Define standard folder name based on your project structure
DATA_DIR = 'data'

# Load Data
# Use os.path.join to create the cross-platform path
data_path = os.path.join(DATA_DIR, 'bike_sales_data.csv')
df = pd.read_csv(data_path) 
spend_cols = [col for col in df.columns if 'spend' in col]

# --- 1. Adstock Transformation Function (Geometric Decay) ---
def geometric_adstock(series, decay_factor):
    """
    Applies Geometric Adstock (carryover effect) to a time series.
    Formula: Adstock_t = Spend_t + decay_factor * Adstock_{t-1}
    """
    # Create an array for the adstocked series, initialized to zero
    adstock_series = np.zeros_like(series, dtype=float)
    
    # Calculate the adstock for each period
    for t in range(len(series)):
        # Adstock is current spend + (decay * previous adstock)
        adstock_series[t] = series[t] + (decay_factor * adstock_series[t-1] if t > 0 else 0)
        
    return adstock_series

# --- 2. Saturation Transformation Function (Simple Log) ---
def log_saturation(series, constant=1):
    """
    Applies a Logarithmic transformation to model diminishing returns (saturation).
    Formula: Log(Spend + C)
    """
    # The '+ constant' handles zero-spend values safely (e.g., Log(0) is undefined)
    return np.log(series + constant)

# --- 3. Apply Transformations to All Spend Channels ---
transformed_df = df.copy()
adstock_decay = 0.5  # Example decay rate: 50% of ad effect carries over to next week
log_constant = 1     # Small constant to handle zero values

for col in spend_cols:
    # 1. Apply Adstock
    adstock_col = f'{col}_adstock_{adstock_decay}'
    transformed_df[adstock_col] = geometric_adstock(transformed_df[col], adstock_decay)
    
    # 2. Apply Saturation (to the Adstocked feature)
    adstock_sat_col = f'{col}_adstock_sat'
    transformed_df[adstock_sat_col] = log_saturation(transformed_df[adstock_col], log_constant)
    
    # Drop the original and intermediate adstock columns to clean up the final model data
    transformed_df = transformed_df.drop(columns=[col, adstock_col])

# Display the new transformed features
print("Transformed Data Head (New Features):")
print(transformed_df.head().to_string(index=False))

# Save the processed data for the final model (saved in the root directory for the next script)
transformed_df.to_csv('bike_sales_data_transformed.csv', index=False)
print("\nSaved: bike_sales_data_transformed.csv with new Adstock & Saturation features.")
