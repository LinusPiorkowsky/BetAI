import os
import pandas as pd
from datetime import datetime, timedelta

# Define seasons and leagues
seasons = ['2024_25']
leagues = ['D1', 'F1', 'E0', 'I1', 'SP1']
timezone = 1  # Adjust for timezone correction

# Function to safely load CSV files
def safe_load_csv(filepath):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        print(f"⚠️ Warning: File {filepath} not found.")
        return pd.DataFrame()  # Return empty DataFrame if file not found

# Load historical match results
historical_data = pd.concat([
    safe_load_csv(f"data/{season}/{league}.csv")
    for season in seasons
    for league in leagues
], ignore_index=True)

### PROCESS HISTORICAL DATA ###

# Keep only matches with valid results
historical_data = historical_data[historical_data['FTR'].isin(['H', 'D', 'A'])].copy()

# Convert 'Date' column to datetime format
historical_data['Date'] = pd.to_datetime(historical_data['Date'], dayfirst=True, errors='coerce')

# Columns to keep
columns_to_keep = ['Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
historical_data = historical_data[columns_to_keep]

# Drop rows with missing values
historical_data.dropna(inplace=True)

# Adjust time with timezone correction
historical_data['Time'] = pd.to_datetime(historical_data['Time'].astype(str), format='%H:%M:%S', errors='coerce') + timedelta(hours=timezone)
historical_data['Time'] = historical_data['Time'].dt.strftime('%H:%M')

# Sort by date and time
historical_data.sort_values(by=["Date", "Time"], inplace=True)

# Add weekday column
historical_data['Weekday'] = historical_data['Date'].dt.day_name()

### PROCESS PREDICTIONS ###

def get_next_result_number():
    """Get the next available result number by checking existing files."""
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    existing_files = [f for f in os.listdir(results_dir) if f.startswith('result_') and f.endswith('.csv')]
    if not existing_files:
        return 1
    
    numbers = [int(f.replace('result_', '').replace('.csv', '')) for f in existing_files]
    return max(numbers) + 1

def compare_predictions_with_results(predictions_df, historical_df):
    """
    Compare prediction data with historical results and generate a comparison CSV.
    
    Parameters:
    predictions_df (pandas.DataFrame): DataFrame containing predictions
    historical_df (pandas.DataFrame): DataFrame containing historical results
    
    Returns:
    pandas.DataFrame: Merged comparison data
    """
    # Convert date columns to datetime if they aren't already
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'], format='%Y-%m-%d')
    historical_df['Date'] = pd.to_datetime(historical_df['Date'], dayfirst=True)
    
    # Merge predictions with historical results
    comparison = pd.merge(
        predictions_df,
        historical_df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']],
        on=['Date', 'HomeTeam', 'AwayTeam'],
        how='left'
    )
    
    # Add evaluation columns
    comparison['Actual_Result'] = comparison['FTR']
    comparison['Prediction_Correct'] = comparison['Prediction'] == comparison['FTR']
    
    # Calculate if double chance predictions were correct
    comparison['DC_1X_Correct'] = comparison.apply(
        lambda x: x['FTR'] in ['H', 'D'] if pd.notna(x['1X_odds']) else None, 
        axis=1
    )
    comparison['DC_X2_Correct'] = comparison.apply(
        lambda x: x['FTR'] in ['D', 'A'] if pd.notna(x['X2_odds']) else None, 
        axis=1
    )
    
    # Format scores
    comparison['Score'] = comparison.apply(
        lambda x: f"{int(x['FTHG'])}:{int(x['FTAG'])}" if pd.notna(x['FTHG']) and pd.notna(x['FTAG']) else None,
        axis=1
    )
    
    # Select and order columns for output
    output_columns = [
        'Div', 'Date', 'Weekday', 'Time', 'HomeTeam', 'AwayTeam',
        'Prediction', 'Actual_Result', 'Score', 'Prediction_Correct',
        'B365H', 'B365D', 'B365A',
        'Prob_H', 'Prob_D', 'Prob_A',
        'double_chance', '1X_odds', 'X2_odds', 'DC_1X_Correct', 'DC_X2_Correct',
        'High_conf', 'High_conf_dc'
    ]
    
    comparison_output = comparison[output_columns]
    
    # Get next result number and save
    next_number = get_next_result_number()
    output_path = f'results/result_{next_number}.csv'
    comparison_output.to_csv(output_path, index=False)
    print(f"Comparison results saved to {output_path}")
    
    return comparison_output

# List all files in the 'predictions' directory
prediction_files = [f for f in os.listdir('predictions') if f.endswith('.csv')]

# Sort the files to get the latest one
prediction_files.sort()

# Get the latest prediction file
latest_prediction_file = prediction_files[-1]

# Read the latest prediction file
latest_predictions = pd.read_csv(f'predictions/{latest_prediction_file}')

# Compare predictions with historical data
comparison_results = compare_predictions_with_results(
    latest_predictions,
    historical_data
)

# Filter for high confidence predictions
high_confidence_results = comparison_results[comparison_results['High_conf'] == 1]
# print("\nHigh Confidence Predictions:")
# print(high_confidence_results[['HomeTeam', 'AwayTeam', 'Prediction', 'Actual_Result', 'Score']])

# Print summary statistics for high confidence predictions
# print("\nHigh Confidence Prediction Results Summary:")
# print(f"Total high confidence matches evaluated: {len(high_confidence_results)}")
# print(f"Correct high confidence predictions: {high_confidence_results['Prediction_Correct'].sum()}")
# print(f"High confidence prediction accuracy: {(high_confidence_results['Prediction_Correct'].mean() * 100):.2f}%")
# print(f"High confidence Double Chance X2 accuracy: {(high_confidence_results['DC_X2_Correct'].mean() * 100):.2f}%")