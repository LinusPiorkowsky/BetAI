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

# Calculate xG based on shots on target and goals
def calculate_xg(row, is_home=True):
    shots = row['HST' if is_home else 'AST']
    goals = row['FTHG' if is_home else 'FTAG']
    conversion_rate = 0.3  # Average conversion rate for shots on target
    return shots * conversion_rate

historical_data['xGH'] = historical_data.apply(lambda x: calculate_xg(x, True), axis=1)
historical_data['xGA'] = historical_data.apply(lambda x: calculate_xg(x, False), axis=1)

# Encoding 'FTR' as numerical values
label_mapping = {'H': 0, 'D': 1, 'A': 2}
historical_data['FTR_encoded'] = historical_data['FTR'].map(label_mapping)

# Prepare fixtures
fixtures['Date'] = pd.to_datetime(fixtures['Date'], dayfirst=True)
fixtures['DateTime'] = pd.to_datetime(fixtures['Date'].astype(str) + ' ' + fixtures['Time'])
fixtures['Time'] = fixtures['DateTime'].dt.time
fixtures.sort_values("Date", inplace=True)

# Calculate league positions
def calculate_league_positions(df):
    positions = {}
    current_season = None
    current_league = None
    league_table = defaultdict(lambda: {'points': 0, 'games': 0, 'gd': 0})
    
    for idx, row in df.iterrows():
        date = row['Date']
        season = date.year if date.month > 6 else date.year - 1
        
        # Reset tables for new season
        if season != current_season or row['Div'] != current_league:
            league_table.clear()
            current_season = season
            current_league = row['Div']
        
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Store current positions before updating
        home_pos = sorted(league_table.items(), key=lambda x: (-x[1]['points'], -x[1]['gd'], -x[1]['games']))
        home_rank = next((i + 1 for i, (team, _) in enumerate(home_pos) if team == home_team), len(league_table) + 1)
        away_rank = next((i + 1 for i, (team, _) in enumerate(home_pos) if team == away_team), len(league_table) + 1)
        
        df.at[idx, 'HomeTeamPosition'] = home_rank
        df.at[idx, 'AwayTeamPosition'] = away_rank
        
        # Update league table after storing positions
        if row['FTR'] == 'H':
            league_table[home_team]['points'] += 3
        elif row['FTR'] == 'A':
            league_table[away_team]['points'] += 3
        else:
            league_table[home_team]['points'] += 1
            league_table[away_team]['points'] += 1
            
        league_table[home_team]['games'] += 1
        league_table[away_team]['games'] += 1
        league_table[home_team]['gd'] += row['FTHG'] - row['FTAG']
        league_table[away_team]['gd'] += row['FTAG'] - row['FTHG']
    
    return df

historical_data = calculate_league_positions(historical_data)

# Enhanced team stats function including last N matches
def team_stats(df, team, home_away, date, n_matches=5):
    is_home = home_away == 'home'
    team_games = df[
        (((df['HomeTeam'] == team) & is_home) | ((df['AwayTeam'] == team) & (not is_home))) &
        (df['Date'] < date)
    ].sort_values('Date', ascending=False).head(n_matches)
    
    if len(team_games) == 0:
        return 0, 0, 0, 0, 0, 0
    
    goals_scored = []
    goals_conceded = []
    xg_for = []
    xg_against = []
    
    for _, game in team_games.iterrows():
        if (is_home and game['HomeTeam'] == team) or (not is_home and game['AwayTeam'] == team):
            goals_scored.append(game['FTHG' if is_home else 'FTAG'])
            goals_conceded.append(game['FTAG' if is_home else 'FTHG'])
            xg_for.append(game['xGH' if is_home else 'xGA'])
            xg_against.append(game['xGA' if is_home else 'xGH'])
    
    recent_goals_scored = np.mean(goals_scored) if goals_scored else 0
    recent_goals_conceded = np.mean(goals_conceded) if goals_conceded else 0
    recent_xg_for = np.mean(xg_for) if xg_for else 0
    recent_xg_against = np.mean(xg_against) if xg_against else 0
    
    # Calculate overall stats
    all_team_games = df[
        (((df['HomeTeam'] == team) & is_home) | ((df['AwayTeam'] == team) & (not is_home))) &
        (df['Date'] < date)
    ]
    
    overall_goals_scored = all_team_games['FTHG' if is_home else 'FTAG'].mean() if len(all_team_games) > 0 else 0
    overall_goals_conceded = all_team_games['FTAG' if is_home else 'FTHG'].mean() if len(all_team_games) > 0 else 0
    
    return (
        np.nan_to_num(overall_goals_scored),
        np.nan_to_num(overall_goals_conceded),
        np.nan_to_num(recent_goals_scored),
        np.nan_to_num(recent_goals_conceded),
        np.nan_to_num(recent_xg_for),
        np.nan_to_num(recent_xg_against)
    )

# Enhanced form calculation with separate home/away form
def add_enhanced_form(df, n_games=5):
    team_home_form = defaultdict(list)
    team_away_form = defaultdict(list)
    
    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        date = row['Date']
        
        # Get recent home and away form
        home_recent = team_home_form[home_team][-n_games:] if team_home_form[home_team] else []
        away_recent = team_away_form[away_team][-n_games:] if team_away_form[away_team] else []
        
        # Calculate separate home and away form
        df.at[index, 'home_form_home'] = sum(home_recent) / len(home_recent) if home_recent else 0
        df.at[index, 'away_form_away'] = sum(away_recent) / len(away_recent) if away_recent else 0
        
        # Update form based on match result
        if row['FTR'] == 'H':
            team_home_form[home_team].append(1)
            team_away_form[away_team].append(0)
        elif row['FTR'] == 'A':
            team_home_form[home_team].append(0)
            team_away_form[away_team].append(1)
        else:
            team_home_form[home_team].append(0.5)
            team_away_form[away_team].append(0.5)
    
    return df

historical_data = add_enhanced_form(historical_data)

# Keep the existing head_to_head function
def head_to_head(df, home_team, away_team, date, n_games=3):
    past_matches = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
                      ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team)) &
                      (df['Date'] < date)]
    recent_matches = past_matches.tail(n_games)
    home_wins = (recent_matches['HomeTeam'] == home_team) & (recent_matches['FTR'] == 'H')
    away_wins = (recent_matches['AwayTeam'] == home_team) & (recent_matches['FTR'] == 'A')
    draws = recent_matches['FTR'] == 'D'
    return home_wins.sum(), away_wins.sum(), draws.sum()

# Enhanced compute_features function
def compute_features(df, reference_df):
    features = [
        'home_goals_scored', 'home_goals_conceded',
        'away_goals_scored', 'away_goals_conceded',
        'recent_home_goals_scored', 'recent_home_goals_conceded',
        'recent_away_goals_scored', 'recent_away_goals_conceded',
        'home_xg_for', 'home_xg_against',
        'away_xg_for', 'away_xg_against',
        'goal_difference',
        'home_form_home', 'away_form_away',
        'head_to_head_home_win', 'head_to_head_away_win', 'head_to_head_draw',
        'HomeTeamPosition', 'AwayTeamPosition'
    ]
    
    for feature in features:
        df[feature] = 0.0
    
    for index, row in df.iterrows():
        (home_goals_scored, home_goals_conceded,
         recent_home_goals_scored, recent_home_goals_conceded,
         home_xg_for, home_xg_against) = team_stats(reference_df, row['HomeTeam'], 'home', row['Date'])
        
        (away_goals_scored, away_goals_conceded,
         recent_away_goals_scored, recent_away_goals_conceded,
         away_xg_for, away_xg_against) = team_stats(reference_df, row['AwayTeam'], 'away', row['Date'])
        
        goal_difference = float(home_goals_scored - away_goals_conceded)
        
        df.at[index, 'home_goals_scored'] = home_goals_scored
        df.at[index, 'home_goals_conceded'] = home_goals_conceded
        df.at[index, 'away_goals_scored'] = away_goals_scored
        df.at[index, 'away_goals_conceded'] = away_goals_conceded
        df.at[index, 'recent_home_goals_scored'] = recent_home_goals_scored
        df.at[index, 'recent_home_goals_conceded'] = recent_home_goals_conceded
        df.at[index, 'recent_away_goals_scored'] = recent_away_goals_scored
        df.at[index, 'recent_away_goals_conceded'] = recent_away_goals_conceded
        df.at[index, 'home_xg_for'] = home_xg_for
        df.at[index, 'home_xg_against'] = home_xg_against
        df.at[index, 'away_xg_for'] = away_xg_for
        df.at[index, 'away_xg_against'] = away_xg_against
        df.at[index, 'goal_difference'] = goal_difference
        
        h2h_home, h2h_away, h2h_draw = head_to_head(reference_df, row['HomeTeam'], row['AwayTeam'], row['Date'])
        df.at[index, 'head_to_head_home_win'] = h2h_home
        df.at[index, 'head_to_head_away_win'] = h2h_away
        df.at[index, 'head_to_head_draw'] = h2h_draw
    
    return df

# Apply enhanced features
historical_data = compute_features(historical_data, historical_data)
fixtures = compute_features(fixtures, historical_data)

# Update feature columns list for model training
feature_columns = [
    'home_goals_scored', 'home_goals_conceded',
    'away_goals_scored', 'away_goals_conceded',
    'recent_home_goals_scored', 'recent_home_goals_conceded',
    'recent_away_goals_scored', 'recent_away_goals_conceded',
    'home_xg_for', 'home_xg_against',
    'away_xg_for', 'away_xg_against',
    'goal_difference',
    'home_form_home', 'away_form_away',
    'head_to_head_home_win', 'head_to_head_away_win', 'head_to_head_draw',
    'HomeTeamPosition', 'AwayTeamPosition'
]

# Rest of your code remains the same from here
X = historical_data[feature_columns]
y = historical_data['FTR_encoded']

# Train/Test Split
split_date = pd.to_datetime('2023-01-01')
train_data = historical_data[historical_data['Date'] < split_date]
test_data = historical_data[historical_data['Date'] >= split_date]

# Continuing from the previous split
X_train = train_data[feature_columns]
y_train = train_data['FTR_encoded']
X_test = test_data[feature_columns]
y_test = test_data['FTR_encoded']

# For prediction on fixtures, use the entire historical data as reference
reference_data = historical_data[historical_data['Date'] < pd.to_datetime('2024-01-01')]
fixtures = compute_features(fixtures, reference_data)

# Feature Scaling
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

# Fit the model on the entire historical data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
best_rf_model.fit(X_scaled, y)

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

# Add a 'High_conf' column based on the specified conditions
fixtures['High_conf'] = ((fixtures['Prob_H'] > 0.7) | (fixtures['Prob_A'] > 0.67)).astype(int)

# Add Weekday
fixtures['Weekday'] = fixtures['Date'].dt.day_name()

# Add timezones
fixtures.loc[:, 'Time'] = (pd.to_datetime(fixtures['Time'].astype(str), format='%H:%M:%S') + pd.Timedelta(hours=timezone)).dt.time
fixtures['Time'] = fixtures['Time'].apply(lambda x: f"{x.hour:02}:{x.minute:02}" if pd.notnull(x) else x)

# Add double chance
fixtures['double_chance'] = fixtures.apply(lambda row: '1X' if row['Prob_H'] + row['Prob_D'] > row['Prob_A'] + row['Prob_D'] else 'X2', axis=1)

# Add double chance odds
fixtures['1X_odds'] = ((fixtures['B365H'] * fixtures['B365D']) / (fixtures['B365H'] + fixtures['B365D'])) 
fixtures['X2_odds'] = ((fixtures['B365D'] * fixtures['B365A']) / (fixtures['B365D'] + fixtures['B365A']))

# Add double chance probabilities
fixtures['1X_prob'] = fixtures['Prob_H'] + fixtures['Prob_D']
fixtures['X2_prob'] = fixtures['Prob_D'] + fixtures['Prob_A']

# round the odds and probabilities
fixtures['1X_odds'] = fixtures['1X_odds'].round(2)
fixtures['X2_odds'] = fixtures['X2_odds'].round(2)
fixtures['1X_prob'] = fixtures['1X_prob'].round(4)
fixtures['X2_prob'] = fixtures['X2_prob'].round(4)

# add high confidence double chance
fixtures['High_conf_dc'] = ((fixtures['1X_prob'] > 0.90) | (fixtures['X2_prob'] > 0.87)).astype(int)

# Prepare the final dataframe for export
predictions_df = fixtures[['Div', 'Date', 'Weekday', 'Time', 'HomeTeam', 'AwayTeam', 'Prediction', 
                          'B365H', 'B365D', 'B365A', 'Prob_H', 'Prob_D', 'Prob_A', 'double_chance',
                          '1X_odds', 'X2_odds', '1X_prob', 'X2_prob', 'High_conf', 'High_conf_dc']]
predictions_df.sort_values(["Date", "Time"], inplace=True)

# Angenommen, best_rf_model ist Ihr trainiertes Random Forest Modell
feature_importances = best_rf_model.feature_importances_
# Erstellen Sie ein DataFrame, um die Merkmale und ihre Gewichtungen anzuzeigen
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importances
})
# Sortieren Sie die Merkmale nach ihrer Wichtigkeit
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
# print(feature_importance_df)

# Display the predictions
print("Predictions:")
predictions = predictions_df[['Div', 'Date', 'Weekday', 'Time', 'HomeTeam', 'AwayTeam', 'Prediction',
                            'B365H', 'B365D', 'B365A', 'Prob_H', 'Prob_D', 'Prob_A', 'High_conf']]
print(predictions[predictions['High_conf'] == 1])

# Display the double chance predictions
print("\nDouble Chance Predictions:")
predictions_dc = predictions_df[['Div', 'Date', 'Weekday', 'Time', 'HomeTeam', 'AwayTeam',
                               'double_chance', '1X_prob', 'X2_prob', '1X_odds', 'X2_odds', 'High_conf', 'High_conf_dc']]
print(predictions_dc[(predictions_dc['High_conf_dc'] == 1) & (predictions_dc['High_conf'] == 1)])

# Create the predictions directory if it doesn't exist
os.makedirs('predictions', exist_ok=True)

# Find the next available filename
i = 1
while os.path.exists(f'predictions/predictions_{i}.csv'):
    i += 1

# Export to CSV with the new filename
predictions_df.to_csv(f'predictions/predictions_{i}.csv', index=False)
print(f"Predictions exported to predictions/predictions_{i}.csv")