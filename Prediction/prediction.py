import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import os

# Define the timezone offset
timezone = 1

# Define seasons and leagues
seasons = ['2019_20', '2020_21', '2021_22', '2022_23', '2023_24', '2024_25']
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

# Encoding 'FTR' as numerical values
label_mapping = {'H': 0, 'D': 1, 'A': 2}
historical_data['FTR_encoded'] = historical_data['FTR'].map(label_mapping)

# Prepare fixtures
fixtures['Date'] = pd.to_datetime(fixtures['Date'], dayfirst=True)
fixtures['DateTime'] = pd.to_datetime(fixtures['Date'].astype(str) + ' ' + fixtures['Time'])
fixtures['Time'] = fixtures['DateTime'].dt.time
fixtures.sort_values("Date", inplace=True)

# Feature Engineering (Team Stats)
def team_stats(df, team, home_away, date):
    is_home = home_away == 'home'
    relevant_games = df[((df['HomeTeam'] == team) & (df['Date'] < date)) if is_home else ((df['AwayTeam'] == team) & (df['Date'] < date))]
    goals_scored = relevant_games['FTHG' if is_home else 'FTAG'].mean()
    goals_conceded = relevant_games['FTAG' if is_home else 'FTHG'].mean()
    return np.nan_to_num(goals_scored), np.nan_to_num(goals_conceded)

# Adding recent form (Last 5 games)
def add_recent_form(df, n_games=5):
    team_form = defaultdict(list)
    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        date = row['Date']
        
        # Recent performance tracking
        home_form = team_form[home_team]
        away_form = team_form[away_team]
        
        # Calculate form for home and away teams
        home_recent = home_form[-n_games:] if len(home_form) > n_games else home_form
        away_recent = away_form[-n_games:] if len(away_form) > n_games else away_form
        
        # Form value calculation
        df.at[index, 'home_form'] = sum(home_recent) / len(home_recent) if home_recent else 0
        df.at[index, 'away_form'] = sum(away_recent) / len(away_recent) if away_recent else 0
        
        # Update team form
        if row['FTR'] == 'H':
            home_form.append(1)  # Win
            away_form.append(0)  # Loss
        elif row['FTR'] == 'A':
            home_form.append(0)  # Loss
            away_form.append(1)  # Win
        else:
            home_form.append(0.5)  # Draw
            away_form.append(0.5)  # Draw
        
        team_form[home_team] = home_form
        team_form[away_team] = away_form

    return df

historical_data = add_recent_form(historical_data)

# Head-to-Head Performance
def head_to_head(df, home_team, away_team, date, n_games=3):
    past_matches = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
                      ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team)) &
                      (df['Date'] < date)]
    recent_matches = past_matches.tail(n_games)
    home_wins = (recent_matches['HomeTeam'] == home_team) & (recent_matches['FTR'] == 'H')
    away_wins = (recent_matches['AwayTeam'] == home_team) & (recent_matches['FTR'] == 'A')
    draws = recent_matches['FTR'] == 'D'
    return home_wins.sum(), away_wins.sum(), draws.sum()

# Add head-to-head results as features
historical_data['head_to_head_home_win'], historical_data['head_to_head_away_win'], historical_data['head_to_head_draw'] = zip(*historical_data.apply(
    lambda row: head_to_head(historical_data, row['HomeTeam'], row['AwayTeam'], row['Date']), axis=1))

# Feature Engineering (Compute Stats)
def compute_features(df, reference_df):
    features = ['home_goals_scored', 'home_goals_conceded', 'away_goals_scored', 'away_goals_conceded', 'goal_difference', 'home_form', 'away_form', 'head_to_head_home_win', 'head_to_head_away_win', 'head_to_head_draw']
    for feature in features:
        df[feature] = 0.0
        
    for index, row in df.iterrows():
        home_goals_scored, home_goals_conceded = team_stats(reference_df, row['HomeTeam'], 'home', row['Date'])
        away_goals_scored, away_goals_conceded = team_stats(reference_df, row['AwayTeam'], 'away', row['Date'])
        
        # Konvertieren Sie die Werte in den richtigen Datentyp
        home_goals_scored = float(home_goals_scored)
        home_goals_conceded = float(home_goals_conceded)
        away_goals_scored = float(away_goals_scored)
        away_goals_conceded = float(away_goals_conceded)
        goal_difference = float(home_goals_scored - away_goals_conceded)
        
        # Setzen Sie die Werte in den DataFrame
        df.at[index, 'home_goals_scored'] = home_goals_scored
        df.at[index, 'home_goals_conceded'] = home_goals_conceded
        df.at[index, 'away_goals_scored'] = away_goals_scored
        df.at[index, 'away_goals_conceded'] = away_goals_conceded
        df.at[index, 'goal_difference'] = goal_difference

    return df

historical_data = compute_features(historical_data, historical_data)
fixtures = compute_features(fixtures, historical_data)

# Prepare Features and Target
feature_columns = ['home_goals_scored', 'home_goals_conceded', 'away_goals_scored', 'away_goals_conceded', 'goal_difference', 'home_form', 'away_form', 'head_to_head_home_win', 'head_to_head_away_win', 'head_to_head_draw']
X = historical_data[feature_columns]
y = historical_data['FTR_encoded']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling (for other models like SVM, not necessary for RandomForest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Parameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4]
}

# Create class weights dictionary
class_weights = {0: 1.0, 1: 1.0, 2: 1.0}

# Set up GridSearchCV with StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=5)

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight=class_weights),
    param_grid,
    cv=stratified_kfold,
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)
best_rf_model = grid_search.best_estimator_

# Model Evaluation
y_pred = best_rf_model.predict(X_test_scaled)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Generate a classification report
class_report = classification_report(y_test, y_pred, target_names=['Home Win', 'Draw', 'Away Win'])
print("Classification Report:")
print(class_report)

# Get predictions and probabilities for fixtures
fixtures_scaled = scaler.transform(fixtures[feature_columns])
fixtures['Predicted Result'] = best_rf_model.predict(fixtures_scaled)

# Get probability scores for each outcome
probabilities = best_rf_model.predict_proba(fixtures_scaled)
fixtures['Prob_H'] = probabilities[:, 0]  # Home win probability
fixtures['Prob_D'] = probabilities[:, 1]  # Draw probability
fixtures['Prob_A'] = probabilities[:, 2]  # Away win probability

# Create a mapping to decode the numerical prediction back to 'H', 'D', 'A'
reverse_label_mapping = {0: 'H', 1: 'D', 2: 'A'}

# Decode the predictions back to 'H', 'D', 'A'
fixtures['Prediction'] = fixtures['Predicted Result'].map(reverse_label_mapping)

fixtures['Prob_H'] = fixtures['Prob_H'].round(4)
fixtures['Prob_D'] = fixtures['Prob_D'].round(4)
fixtures['Prob_A'] = fixtures['Prob_A'].round(4)

# Add a 'High Confidence Bet' column based on the specified conditions
fixtures['High Confidence Bet'] = ((fixtures['Prob_H'] > 0.65) | (fixtures['Prob_A'] > 0.62)).astype(int)

#Add Weekday
fixtures['Weekday'] = fixtures['Date'].dt.day_name()

#Add timezones
fixtures['Time'] = (pd.to_datetime(fixtures['Time'].astype(str)) + pd.Timedelta(hours=timezone)).dt.time
fixtures['Time'] = fixtures['Time'].apply(lambda x: f"{x.hour:02}:{x.minute:02}" if pd.notnull(x) else x)

# Prepare the final dataframe for export
predictions_df = fixtures[['Div', 'Date', 'Weekday' ,'Time', 'HomeTeam', 'AwayTeam', 'Prediction', 'B365H', 'B365D', 'B365A', 'Prob_H', 'Prob_D', 'Prob_A', 'Best Bet']]
predictions_df.sort_values(["Date", "Time"], inplace=True)
print(predictions_df[predictions_df['Best Bet'] == 1])

# Create the predictions directory if it doesn't exist
os.makedirs('predictions', exist_ok=True)

# Find the next available filename
i = 1
while os.path.exists(f'predictions/predictions_{i}.csv'):
    i += 1

# Export to CSV with the new filename
predictions_df.to_csv(f'predictions/predictions_{i}.csv', index=False)
print(f"Predictions exported to data/predictions/predictions_{i}.csv")
