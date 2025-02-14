import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define seasons and leagues
seasons = ['2017_18', '2018_19', '2019_20', '2020_21', '2021_22', '2022_23', '2023_24', '2024_25']
leagues = ['D1', 'F1', 'E0', 'I1', 'SP1']

# Load historical data
historical_data = pd.concat([
    pd.read_csv(f"data/{season}/{league}.csv")
    for season in seasons
    for league in leagues
])

#load fixtures
fixtures = pd.read_csv("data/fixtures/fixtures.csv")

### historical data

# Keep only valid match results
historical_data = historical_data[historical_data['FTR'].isin(['H', 'D', 'A'])].copy()

# Convert 'Date' column to datetime format
historical_data['Date'] = pd.to_datetime(historical_data['Date'], dayfirst=True)

# Columns to keep
columns_to_keep = [
    'Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
    'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'B365>2.5', 'B365<2.5', 
    'B365H', 'B365D', 'B365A'
]

# Keep only specified columns
historical_data = historical_data[columns_to_keep]

# Drop rows with any missing values
historical_data.dropna(inplace=True)

# Reset the index
historical_data.reset_index(drop=True, inplace=True)

#date sorted
historical_data.sort_values("Date", inplace=True)

### fixtures

# Convert 'Date' column to datetime format
fixtures['Date'] = pd.to_datetime(fixtures['Date'], dayfirst=True)

#convert time to time format
fixtures['DateTime'] = pd.to_datetime(fixtures['Date'].astype(str) + ' ' + fixtures['Time'])

fixtures['Time'] = fixtures['DateTime'].dt.time

# date sorted
fixtures.sort_values("Date", inplace=True)