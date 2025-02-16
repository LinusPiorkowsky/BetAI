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
seasons = ['2017_18', '2018_19', '2019_20', '2020_21', '2021_22', '2022_23', '2023_24', '2024_25']
leagues = ['D1', 'F1', 'E0', 'I1', 'SP1']

def load_and_prepare_data():
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

    return historical_data, fixtures

def calculate_team_stats(df, team, is_home, date, n_games=5):
    """Calculate team statistics for recent games"""
    team_matches = df[
        ((df['HomeTeam'] == team) & (df['Date'] < date)) if is_home 
        else ((df['AwayTeam'] == team) & (df['Date'] < date))
    ].tail(n_games)
    
    if len(team_matches) == 0:
        return {
            'goals_scored': 0,
            'goals_conceded': 0,
            'shots': 0,
            'shots_target': 0,
            'corners': 0,
            'win_rate': 0
        }
    
    cols = ['FTHG', 'FTAG', 'HS', 'HST', 'HC'] if is_home else ['FTAG', 'FTHG', 'AS', 'AST', 'AC']
    stats = team_matches[cols].mean()
    
    wins = team_matches['FTR'].apply(lambda x: 1 if (x == 'H' and is_home) or (x == 'A' and not is_home) else 0).mean()
    
    return {
        'goals_scored': stats.iloc[0],
        'goals_conceded': stats.iloc[1],
        'shots': stats.iloc[2],
        'shots_target': stats.iloc[3],
        'corners': stats.iloc[4],
        'win_rate': wins
    }

def calculate_h2h_stats(df, home_team, away_team, date, n_games=3):
    """Calculate head-to-head statistics"""
    h2h_matches = df[
        (((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
         ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))) &
        (df['Date'] < date)
    ].tail(n_games)
    
    if len(h2h_matches) == 0:
        return {
            'home_wins': 0,
            'away_wins': 0,
            'draws': 0,
            'avg_goals': 0,
            'goal_diff': 0
        }
    
    def calculate_goal_diff(row):
        if row['HomeTeam'] == home_team:
            return row['FTHG'] - row['FTAG']
        return row['FTAG'] - row['FTHG']
    
    home_wins = sum((h2h_matches['HomeTeam'] == home_team) & (h2h_matches['FTR'] == 'H')) + \
                sum((h2h_matches['AwayTeam'] == home_team) & (h2h_matches['FTR'] == 'A'))
    away_wins = sum((h2h_matches['HomeTeam'] == away_team) & (h2h_matches['FTR'] == 'H')) + \
                sum((h2h_matches['AwayTeam'] == away_team) & (h2h_matches['FTR'] == 'A'))
    draws = sum(h2h_matches['FTR'] == 'D')
    
    return {
        'home_wins': home_wins,
        'away_wins': away_wins,
        'draws': draws,
        'avg_goals': (h2h_matches['FTHG'] + h2h_matches['FTAG']).mean(),
        'goal_diff': h2h_matches.apply(calculate_goal_diff, axis=1).mean()
    }

def calculate_form(df, team, date, n_games=5):
    """Calculate team form based on recent results"""
    recent_matches = df[
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
        (df['Date'] < date)
    ].tail(n_games)
    
    if len(recent_matches) == 0:
        return 0
    
    form_points = []
    for _, match in recent_matches.iterrows():
        if match['HomeTeam'] == team:
            points = 3 if match['FTR'] == 'H' else (1 if match['FTR'] == 'D' else 0)
        else:
            points = 3 if match['FTR'] == 'A' else (1 if match['FTR'] == 'D' else 0)
        form_points.append(points)
    
    return sum(form_points) / (len(form_points) * 3)  # Normalize to 0-1 range

def prepare_features(df, reference_df=None):
    """Prepare all features for the model"""
    if reference_df is None:
        reference_df = df.copy()
    
    features = pd.DataFrame(index=df.index)
    
    for idx, row in df.iterrows():
        # Team stats
        home_stats = calculate_team_stats(reference_df, row['HomeTeam'], True, row['Date'])
        away_stats = calculate_team_stats(reference_df, row['AwayTeam'], False, row['Date'])
        h2h_stats = calculate_h2h_stats(reference_df, row['HomeTeam'], row['AwayTeam'], row['Date'])
        
        # Form
        home_form = calculate_form(reference_df, row['HomeTeam'], row['Date'])
        away_form = calculate_form(reference_df, row['AwayTeam'], row['Date'])
        
        # Combine all features
        features.loc[idx, 'home_goals_scored'] = home_stats['goals_scored']
        features.loc[idx, 'home_goals_conceded'] = home_stats['goals_conceded']
        features.loc[idx, 'home_shots'] = home_stats['shots']
        features.loc[idx, 'home_shots_target'] = home_stats['shots_target']
        features.loc[idx, 'home_corners'] = home_stats['corners']
        features.loc[idx, 'home_win_rate'] = home_stats['win_rate']
        
        features.loc[idx, 'away_goals_scored'] = away_stats['goals_scored']
        features.loc[idx, 'away_goals_conceded'] = away_stats['goals_conceded']
        features.loc[idx, 'away_shots'] = away_stats['shots']
        features.loc[idx, 'away_shots_target'] = away_stats['shots_target']
        features.loc[idx, 'away_corners'] = away_stats['corners']
        features.loc[idx, 'away_win_rate'] = away_stats['win_rate']
        
        features.loc[idx, 'h2h_home_wins'] = h2h_stats['home_wins']
        features.loc[idx, 'h2h_away_wins'] = h2h_stats['away_wins']
        features.loc[idx, 'h2h_draws'] = h2h_stats['draws']
        features.loc[idx, 'h2h_avg_goals'] = h2h_stats['avg_goals']
        features.loc[idx, 'h2h_goal_diff'] = h2h_stats['goal_diff']
        
        features.loc[idx, 'home_form'] = home_form
        features.loc[idx, 'away_form'] = away_form
    
    return pd.concat([df, features], axis=1)

def main():
    print("Loading data...")
    historical_data, fixtures = load_and_prepare_data()
    
    print("Preparing features...")
    # Encode target variable
    historical_data['FTR_double_chance'] = historical_data['FTR'].replace({'H': 'H+D', 'D': 'H+D', 'A': 'A+D'})
    historical_data['FTR_encoded'] = historical_data['FTR_double_chance'].map({'H+D': 0, 'A+D': 1})
    
    # Prepare features
    historical_data = prepare_features(historical_data)
    fixtures = prepare_features(fixtures, historical_data)
    
    feature_columns = [
        'home_goals_scored', 'home_goals_conceded', 'home_shots', 'home_shots_target',
        'home_corners', 'home_win_rate', 'away_goals_scored', 'away_goals_conceded',
        'away_shots', 'away_shots_target', 'away_corners', 'away_win_rate',
        'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_avg_goals',
        'h2h_goal_diff', 'home_form', 'away_form'
    ]
    
    print("Training model...")
    X = historical_data[feature_columns]
    y = historical_data['FTR_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [None, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid,
        cv=StratifiedKFold(n_splits=5),
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    
    print("\nEvaluating model...")
    y_pred = best_model.predict(X_test_scaled)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Home or Draw', 'Away or Draw']))
    
    print("\nGenerating predictions...")
    fixtures_scaled = scaler.transform(fixtures[feature_columns])
    probabilities = best_model.predict_proba(fixtures_scaled)
    
    fixtures['Prob_1X'] = probabilities[:, 0].round(4)
    fixtures['Prob_X2'] = probabilities[:, 1].round(4)
    fixtures['Prediction'] = best_model.predict(fixtures_scaled)
    fixtures['Prediction'] = fixtures['Prediction'].map({0: '1X', 1: 'X2'})
    
    # Calculate double chance odds
    fixtures['Odd_1X'] = (1 / (1/fixtures['B365H'] + 1/fixtures['B365D'])).round(2)
    fixtures['Odd_X2'] = (1 / (1/fixtures['B365A'] + 1/fixtures['B365D'])).round(2)
    
    fixtures['High_Confidence'] = ((fixtures['Prob_1X'] > 0.75) | (fixtures['Prob_X2'] > 0.75)).astype(int)

    #Add Weekday
    fixtures['Weekday'] = fixtures['Date'].dt.day_name()

    #Add timezones
    fixtures['Time'] = (pd.to_datetime(fixtures['Time'].astype(str)) + pd.Timedelta(hours=timezone)).dt.time
    fixtures['Time'] = fixtures['Time'].apply(lambda x: f"{x.hour:02}:{x.minute:02}" if pd.notnull(x) else x)
    
    output_columns = [
        'Div', 'Date','Weekday', 'Time', 'HomeTeam', 'AwayTeam', 'Prediction',
        'Prob_1X', 'Prob_X2', 'Odd_1X', 'Odd_X2', 'High_Confidence'
    ]
    
    predictions_df = fixtures[output_columns].copy()
    predictions_df.sort_values(['Date', 'Time'], inplace=True)
    
    os.makedirs('double_chance', exist_ok=True)
    i = 1
    while os.path.exists(f'double_chance/predictions_{i}.csv'):
        i += 1
    
    predictions_df.to_csv(f'double_chance/predictions_{i}.csv', index=False)
    print(f"\nPredictions exported to double_chance/predictions_{i}.csv")
    
    print("\nHigh Confidence Predictions:")
    print(predictions_df[predictions_df['High_Confidence'] == 1])

if __name__ == "__main__":
    main()