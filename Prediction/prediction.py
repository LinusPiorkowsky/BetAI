import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from collections import defaultdict

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

# Encode teams
team_encoder = LabelEncoder()
all_teams = pd.concat([historical_data["HomeTeam"], historical_data["AwayTeam"]]).unique()
team_encoder.fit(all_teams)
historical_data["HomeTeam_encoded"] = team_encoder.transform(historical_data["HomeTeam"])
historical_data["AwayTeam_encoded"] = team_encoder.transform(historical_data["AwayTeam"])

# Expected Goals Feature
historical_data["xG"] = historical_data["HST"].rolling(5).mean() * (historical_data["FTHG"] / historical_data["HST"].replace(0, 1)).rolling(5).mean()
historical_data["xGA"] = historical_data["AST"].rolling(5).mean() * (historical_data["FTAG"] / historical_data["AST"].replace(0, 1)).rolling(5).mean()

# Feature Engineering (Team Stats)
def team_stats(df, team, home_away, date):
    is_home = home_away == 'home'
    relevant_games = df[((df['HomeTeam'] == team) & (df['Date'] < date)) if is_home else ((df['AwayTeam'] == team) & (df['Date'] < date))]
    
    goals_scored = relevant_games['FTHG' if is_home else 'FTAG'].mean()
    goals_conceded = relevant_games['FTAG' if is_home else 'FTHG'].mean()
    shots = relevant_games['HS' if is_home else 'AS'].mean()
    shots_target = relevant_games['HST' if is_home else 'AST'].mean()
    
    return (np.nan_to_num(x) for x in [goals_scored, goals_conceded, shots, shots_target])

# Adding recent form with enhanced metrics
def add_recent_form(df, n_games=5):
    team_form = defaultdict(list)
    team_points = defaultdict(list)
    
    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Recent form tracking
        home_form = team_form[home_team][-n_games:] if len(team_form[home_team]) > n_games else team_form[home_team]
        away_form = team_form[away_team][-n_games:] if len(team_form[away_team]) > n_games else team_form[away_team]
        
        # Points tracking
        home_points = team_points[home_team][-n_games:] if len(team_points[home_team]) > n_games else team_points[home_team]
        away_points = team_points[away_team][-n_games:] if len(team_points[away_team]) > n_games else team_points[away_team]
        
        # Calculate metrics
        df.at[index, 'home_form'] = sum(home_form) / len(home_form) if home_form else 0
        df.at[index, 'away_form'] = sum(away_form) / len(away_form) if away_form else 0
        df.at[index, 'home_points_trend'] = sum(home_points) / len(home_points) if home_points else 0
        df.at[index, 'away_points_trend'] = sum(away_points) / len(away_points) if away_points else 0
        
        # Update form and points based on result
        if row['FTR'] == 'H':
            home_form.append(1)
            away_form.append(0)
            team_points[home_team].append(3)
            team_points[away_team].append(0)
        elif row['FTR'] == 'A':
            home_form.append(0)
            away_form.append(1)
            team_points[home_team].append(0)
            team_points[away_team].append(3)
        else:
            home_form.append(0.5)
            away_form.append(0.5)
            team_points[home_team].append(1)
            team_points[away_team].append(1)
        
        team_form[home_team] = home_form
        team_form[away_team] = away_form
    
    return df

historical_data = add_recent_form(historical_data)

# Compute all features
def compute_features(df, reference_df):
    features = [
        'home_goals_scored', 'home_goals_conceded', 'away_goals_scored', 'away_goals_conceded',
        'home_shots', 'home_shots_target', 'away_shots', 'away_shots_target',
        'goal_difference', 'home_form', 'away_form', 'home_points_trend', 'away_points_trend'
    ]
    
    for feature in features:
        df[feature] = 0
        
    for index, row in df.iterrows():
        home_stats = team_stats(reference_df, row['HomeTeam'], 'home', row['Date'])
        away_stats = team_stats(reference_df, row['AwayTeam'], 'away', row['Date'])
        
        (home_goals_scored, home_goals_conceded, home_shots, home_shots_target) = home_stats
        (away_goals_scored, away_goals_conceded, away_shots, away_shots_target) = away_stats
        
        df.at[index, 'home_goals_scored'] = home_goals_scored
        df.at[index, 'home_goals_conceded'] = home_goals_conceded
        df.at[index, 'away_goals_scored'] = away_goals_scored
        df.at[index, 'away_goals_conceded'] = away_goals_conceded
        df.at[index, 'home_shots'] = home_shots
        df.at[index, 'home_shots_target'] = home_shots_target
        df.at[index, 'away_shots'] = away_shots
        df.at[index, 'away_shots_target'] = away_shots_target
        df.at[index, 'goal_difference'] = home_goals_scored - away_goals_scored
        
    return df

historical_data = compute_features(historical_data, historical_data)
fixtures = compute_features(fixtures, historical_data)

# Prepare Features and Target
feature_columns = [
    'HomeTeam_encoded', 'AwayTeam_encoded',
    'home_goals_scored', 'home_goals_conceded', 
    'away_goals_scored', 'away_goals_conceded',
    'home_shots', 'home_shots_target',
    'away_shots', 'away_shots_target',
    'goal_difference', 'home_form', 'away_form',
    'home_points_trend', 'away_points_trend',
    'xG', 'xGA'
]

X = historical_data[feature_columns]
y = historical_data['FTR']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [6, 8, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': [None, {0: 1.0, 1: 1.5, 2: 1.0}]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
best_rf_model = grid_search.best_estimator_

# Model Evaluation
y_pred = best_rf_model.predict(X_test_scaled)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation scores
cv_scores = cross_val_score(best_rf_model, X_train_scaled, y_train, cv=5)
print(f"Cross-validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean()}")

# Prepare fixtures for prediction
fixtures["HomeTeam_encoded"] = team_encoder.transform(fixtures["HomeTeam"])
fixtures["AwayTeam_encoded"] = team_encoder.transform(fixtures["AwayTeam"])

# Apply model to fixtures and print predictions
fixtures['Predicted Result'] = best_rf_model.predict(scaler.transform(fixtures[feature_columns]))
print(fixtures[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'Predicted Result']])