import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from collections import defaultdict
from imblearn.over_sampling import SMOTE  # For handling class imbalance
from sklearn.feature_selection import RFE

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

# Feature Engineering (Compute Stats)
def compute_features(df, reference_df):
    features = ['home_goals_scored', 'home_goals_conceded', 'away_goals_scored', 'away_goals_conceded', 'goal_difference', 'home_form', 'away_form']
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

historical_data = compute_features(historical_data, historical_data)
fixtures = compute_features(fixtures, historical_data)

# Prepare Features and Target
feature_columns = ['home_goals_scored', 'home_goals_conceded', 'away_goals_scored', 'away_goals_conceded', 'goal_difference', 'home_form', 'away_form']
X = historical_data[feature_columns]
y = historical_data['FTR']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle Class Imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Hyperparameter Tuning with GridSearchCV (extended parameter search)
param_grid = {
    'n_estimators': [200, 300, 500, 1000],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)
best_rf_model = grid_search.best_estimator_

# Model Evaluation
y_pred = best_rf_model.predict(X_test_scaled)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Compute ROC AUC score for each class
roc_auc = roc_auc_score(y_test, best_rf_model.predict_proba(X_test_scaled), multi_class='ovr')
print(f"ROC AUC Score: {roc_auc}")

# Cross-validation
cv_scores = cross_val_score(best_rf_model, X_train_resampled, y_train_resampled, cv=5)
print(f"Cross-validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean()}")

# Feature Importance (Optional)
feature_importances = pd.DataFrame(best_rf_model.feature_importances_,
                                   index=feature_columns, columns=["Importance"]).sort_values("Importance", ascending=False)
print("Feature Importance:")
print(feature_importances)

# Feature Selection (Recursive Feature Elimination)
rfe = RFE(best_rf_model, n_features_to_select=5)
rfe = rfe.fit(X_train_resampled, y_train_resampled)
selected_features = [f for f, s in zip(feature_columns, rfe.support_) if s]
print(f"Selected Features after RFE: {selected_features}")

# Apply model to fixtures and print predictions
fixtures['Predicted Result'] = best_rf_model.predict(scaler.transform(fixtures[selected_features]))
print(fixtures[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'Predicted Result']])
