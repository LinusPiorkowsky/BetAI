import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Define seasons and leagues
seasons = ['2017_18', '2018_19', '2019_20', '2020_21', '2021_22', '2022_23', '2023_24', '2024_25']
leagues = ['D1', 'F1', 'E0', 'I1', 'SP1']

# Load historical data
historical_data = pd.concat([
    pd.read_csv(f"data/{season}/{league}.csv")
    for season in seasons
    for league in leagues
])

# Load fixtures
fixtures = pd.read_csv("data/fixtures/fixtures.csv")

# Prepare historical data
historical_data = historical_data[historical_data['FTR'].isin(['H', 'D', 'A'])]
historical_data['Date'] = pd.to_datetime(historical_data['Date'], dayfirst=True)
columns_to_keep = [
    'Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
    'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'B365>2.5', 'B365<2.5',
    'B365H', 'B365D', 'B365A'
]
historical_data = historical_data[columns_to_keep].dropna().reset_index(drop=True)
historical_data.sort_values("Date", inplace=True)

# Prepare fixtures
fixtures['Date'] = pd.to_datetime(fixtures['Date'], dayfirst=True)
fixtures['DateTime'] = pd.to_datetime(fixtures['Date'].astype(str) + ' ' + fixtures['Time'])
fixtures['Time'] = fixtures['DateTime'].dt.time
fixtures.sort_values("Date", inplace=True)

# Feature Engineering
def team_stats(df, team, home_away, date):
    is_home = home_away == 'home'
    relevant_games = df[((df['HomeTeam'] == team) & (df['Date'] < date)) if is_home else ((df['AwayTeam'] == team) & (df['Date'] < date))]
    goals_scored = relevant_games['FTHG' if is_home else 'FTAG'].mean()
    goals_conceded = relevant_games['FTAG' if is_home else 'FTHG'].mean()
    return np.nan_to_num(goals_scored), np.nan_to_num(goals_conceded)

def compute_features(df, reference_df):
    features = ['home_goals_scored', 'home_goals_conceded', 'away_goals_scored', 'away_goals_conceded', 'goal_difference']
    for feature in features:
        df[feature] = 0
        
    for index, row in df.iterrows():
        home_goals_scored, home_goals_conceded = team_stats(reference_df, row['HomeTeam'], 'home', row['Date'])
        away_goals_scored, away_goals_conceded = team_stats(reference_df, row['AwayTeam'], 'away', row['Date'])
        df.at[index, 'home_goals_scored'] = home_goals_scored
        df.at[index, 'home_goals_conceded'] = home_goals_conceded
        df.at[index, 'away_goals_scored'] = away_goals_scored
        df.at[index, 'away_goals_conceded'] = away_goals_conceded
        df.at[index, 'goal_difference'] = home_goals_scored - away_goals_conceded
    
    return df

# Compute features for historical and fixtures data
historical_data = compute_features(historical_data, historical_data)
fixtures = compute_features(fixtures, historical_data)

# Model preparation
feature_columns = ['home_goals_scored', 'home_goals_conceded', 'away_goals_scored', 'away_goals_conceded', 'goal_difference']
X = historical_data[feature_columns]
y = historical_data['FTR']

# Model training and evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Apply model to fixtures and print predictions
fixtures['Predicted Result'] = rf_model.predict(fixtures[feature_columns])
print(fixtures[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'Predicted Result']])
