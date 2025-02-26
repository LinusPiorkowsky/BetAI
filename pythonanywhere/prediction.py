#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import io
import requests
import pandas as pd
import numpy as np
import zipfile
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# GLOBAL CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = "/home/MachineLearningBets/BetAI"

FIXTURES_DIR = os.path.join(BASE_DIR, "data", "fixtures")
SAVE_DIR     = os.path.join(BASE_DIR, "data", "2024_25")  # Where fresh league CSVs are stored
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")

ALLOWED_LEAGUES = {
    "D1.csv","D2.csv","E0.csv","E1.csv","F1.csv","F2.csv",
    "I1.csv","I2.csv","SP1.csv","SP2.csv"
}

# Example Football-Data URL for the 2024/25 season
DATASET_URL = "https://www.football-data.co.uk/mmz4281/2425/data.zip"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
os.makedirs(FIXTURES_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define seasons and leagues for historical data
seasons = ['2019_20', '2020_21', '2021_22', '2022_23', '2023_24', '2024_25']
leagues = ['D1', 'F1', 'E0', 'I1', 'SP1']

# Timezone offset
timezone = 1

# ---------------------------------------------------------------------------
# 1) DOWNLOAD FIXTURES
# ---------------------------------------------------------------------------
def download_fixtures(url, filename):
    """
    Lädt die Spielansetzungen als CSV herunter, filtert nur die gewünschten Ligen (['D1','E0','F1','I1','SP1'])
    und speichert sie nach data/fixtures/.
    """
    filepath = os.path.join(FIXTURES_DIR, filename)

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        if not response.content:
            print("❌ Fehler: Die heruntergeladene Fixtures-Datei ist leer.")
            return

        df_fixtures = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig")
        df_fixtures.columns = df_fixtures.columns.str.replace("ï»¿", "", regex=True)

        if "Div" not in df_fixtures.columns:
            raise ValueError(f"❌ Fehler: Spalte 'Div' fehlt weiterhin. Columns: {df_fixtures.columns.tolist()}")

        allowed_leagues = ["D1", "E0", "F1", "I1", "SP1"]
        df_fixtures = df_fixtures[df_fixtures["Div"].isin(allowed_leagues)]

        columns_to_keep = ["Div","Date","Time","HomeTeam","AwayTeam","B365H","B365D","B365A","B365>2.5","B365<2.5"]
        df_fixtures = df_fixtures[columns_to_keep]

        df_fixtures.to_csv(filepath, index=False)
        print(f"✅ Fixtures erfolgreich gefiltert und gespeichert: {filepath}")

    except requests.RequestException as e:
        print(f"❌ Fehler beim Herunterladen der Spielansetzungen: {e}")
    except Exception as e:
        print(f"❌ Fehler bei der Verarbeitung der Fixtures: {e}")

# ---------------------------------------------------------------------------
# 2) DOWNLOAD ZIP & EXTRACT (OPTIONAL)
# ---------------------------------------------------------------------------
def download_dataset(url: str, zip_filename: str) -> None:
    """
    Download a ZIP from `url`, extract only ALLOWED_LEAGUES into SAVE_DIR.
    """
    temp_zip_path = os.path.join(SAVE_DIR, zip_filename)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        if not resp.content:
            print("❌ The downloaded dataset is empty.")
            return

        with open(temp_zip_path, 'wb') as f:
            f.write(resp.content)

        with zipfile.ZipFile(temp_zip_path, 'r') as zf:
            for member in zf.namelist():
                if any(member.endswith(league) for league in ALLOWED_LEAGUES):
                    zf.extract(member, SAVE_DIR)

        os.remove(temp_zip_path)
        print(f"✅ Downloaded & extracted dataset into: {SAVE_DIR}")

    except requests.RequestException as e:
        print(f"❌ Error downloading dataset: {e}")
    except zipfile.BadZipFile:
        print("❌ Error: Not a valid ZIP file.")

# ---------------------------------------------------------------------------
# 3) READ EXISTING PREDICTIONS (predictions_*.csv)
# ---------------------------------------------------------------------------
def read_all_predictions(pred_dir: str) -> pd.DataFrame:
    """
    Reads all 'predictions_(N).csv' into a single DataFrame to avoid duplicates.
    """
    all_preds = sorted(
        f for f in os.listdir(pred_dir)
        if re.match(r"predictions_(\d+)\.csv$", f)
    )
    if not all_preds:
        return pd.DataFrame(columns=["Date","HomeTeam","AwayTeam"])

    df_list = []
    for fname in all_preds:
        path = os.path.join(pred_dir, fname)
        df = pd.read_csv(path, parse_dates=["Date"], dayfirst=True, encoding="utf-8")
        df_list.append(df)

    combined_pred = pd.concat(df_list, ignore_index=True).drop_duplicates(subset=["Date","HomeTeam","AwayTeam"])
    return combined_pred

# ---------------------------------------------------------------------------
# 4) MAIN
# ---------------------------------------------------------------------------
def main():
    # (A) Download new fixtures + filter them
    download_fixtures(FIXTURES_URL, "fixtures.csv")

    # (B) Download new league data + extract (optional). Comment out if not needed.
    download_dataset(DATASET_URL, "freshdata.zip")

    # (C) Read historical data from multiple seasons
    seasons = ['2019_20','2020_21','2021_22','2022_23','2023_24','2024_25']
    leagues = ['D1','F1','E0','I1','SP1']

    historical_df_list = []
    for season in seasons:
        for league in leagues:
            path = f"{BASE_DIR}/data/{season}/{league}.csv"
            if os.path.exists(path):
                df_hist = pd.read_csv(path)
                historical_df_list.append(df_hist)
            else:
                print(f"⚠️ Warning: {path} not found.")

    if not historical_df_list:
        print("❌ No historical files found. Exiting.")
        return

    historical_data = pd.concat(historical_df_list, ignore_index=True)
    
    # (D) Load the new fixtures we just downloaded
    fixtures_path = os.path.join(FIXTURES_DIR, "fixtures.csv")
    if not os.path.exists(fixtures_path):
        print(f"❌ Fixtures file not found at {fixtures_path}, aborting.")
        return

    fixtures = pd.read_csv(fixtures_path)

    # (E) Check for duplicates in existing predictions
    #     read all predictions_{N}.csv
    existing_predictions = read_all_predictions(PREDICTIONS_DIR)

    # Merge to see which fixtures are new
    # We'll treat (Date, HomeTeam, AwayTeam) as unique
    if not existing_predictions.empty:
        exist_idx = existing_predictions.set_index(["Date","HomeTeam","AwayTeam"])
        fix_idx   = fixtures.set_index(["Date","HomeTeam","AwayTeam"])
        duplicates_mask = fix_idx.index.isin(exist_idx.index)
        new_fixtures = fixtures[~duplicates_mask].reset_index(drop=True)
        if new_fixtures.empty:
            print("✅ All these fixtures have already been predicted. Nothing new to do.")
            return
        else:
            fixtures = new_fixtures
            print(f"✅ {len(fixtures)} new fixtures to predict.")
    else:
        print("✅ No existing predictions found, so all fixtures are new.")

    # (F) The rest of your model training + predictions code:
    #     (no changes except we use 'fixtures' now, which is guaranteed to have only new matches)

    # Prepare historical data
    historical_data = historical_data[historical_data['FTR'].isin(['H', 'D', 'A'])]
    historical_data['Date'] = pd.to_datetime(historical_data['Date'], dayfirst=True)
    columns_to_keep = [
        'Div','Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR',
        'HS','AS','HST','AST','HC','AC','B365>2.5','B365<2.5',
        'B365H','B365D','B365A'
    ]
    historical_data = historical_data[columns_to_keep].dropna().reset_index(drop=True)
    historical_data.sort_values("Date", inplace=True)

    # xG
    def calculate_xg(row, is_home=True):
        shots = row['HST' if is_home else 'AST']
        goals = row['FTHG' if is_home else 'FTAG']
        conversion_rate = 0.3  # arbitrary average conversion rate
        return shots * conversion_rate

    historical_data['xGH'] = historical_data.apply(lambda x: calculate_xg(x, True), axis=1)
    historical_data['xGA'] = historical_data.apply(lambda x: calculate_xg(x, False), axis=1)

    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    historical_data['FTR_encoded'] = historical_data['FTR'].map(label_mapping)

    # Prepare fixtures similarly
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

            if season != current_season or row['Div'] != current_league:
                league_table.clear()
                current_season = season
                current_league = row['Div']

            home_team = row['HomeTeam']
            away_team = row['AwayTeam']

            # Store current positions
            sorted_table = sorted(league_table.items(), key=lambda x: (-x[1]['points'], -x[1]['gd'], -x[1]['games']))
            home_rank = next((i+1 for i,(team,_) in enumerate(sorted_table) if team == home_team), len(league_table)+1)
            away_rank = next((i+1 for i,(team,_) in enumerate(sorted_table) if team == away_team), len(league_table)+1)

            df.at[idx,'HomeTeamPosition'] = home_rank
            df.at[idx,'AwayTeamPosition'] = away_rank

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

    # Enhanced form
    def add_enhanced_form(df, n_games=5):
        team_home_form = defaultdict(list)
        team_away_form = defaultdict(list)

        for index, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            # get last n games form
            home_recent = team_home_form[home_team][-n_games:] if team_home_form[home_team] else []
            away_recent = team_away_form[away_team][-n_games:] if team_away_form[away_team] else []

            df.at[index, 'home_form_home'] = sum(home_recent)/len(home_recent) if home_recent else 0
            df.at[index, 'away_form_away'] = sum(away_recent)/len(away_recent) if away_recent else 0

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

    # Head to head
    def head_to_head(df, home_team, away_team, date, n_games=3):
        past_matches = df[(
            ((df['HomeTeam']==home_team)&(df['AwayTeam']==away_team))|
            ((df['HomeTeam']==away_team)&(df['AwayTeam']==home_team))
        ) & (df['Date']<date)].tail(n_games)
        home_wins = ((past_matches['HomeTeam']==home_team)&(past_matches['FTR']=='H')).sum()
        away_wins = ((past_matches['AwayTeam']==home_team)&(past_matches['FTR']=='A')).sum()
        draws     = (past_matches['FTR']=='D').sum()
        return (home_wins, away_wins, draws)

    def team_stats(df, team, home_away, date, n_matches=5):
        is_home = (home_away=='home')
        team_games = df[
            (((df['HomeTeam']==team)&is_home)|((df['AwayTeam']==team)&(not is_home))) & (df['Date']<date)
        ].sort_values('Date', ascending=False).head(n_matches)

        if len(team_games)==0:
            return 0,0,0,0,0,0

        goals_scored = []
        goals_conceded = []
        xg_for = []
        xg_against = []

        for _,game in team_games.iterrows():
            if (is_home and game['HomeTeam']==team) or (not is_home and game['AwayTeam']==team):
                goals_scored.append(game['FTHG'] if is_home else game['FTAG'])
                goals_conceded.append(game['FTAG'] if is_home else game['FTHG'])
                xg_for.append(game['xGH'] if is_home else game['xGA'])
                xg_against.append(game['xGA'] if is_home else game['xGH'])

        recent_goals_scored    = np.mean(goals_scored) if goals_scored else 0
        recent_goals_conceded  = np.mean(goals_conceded) if goals_conceded else 0
        recent_xg_for          = np.mean(xg_for) if xg_for else 0
        recent_xg_against      = np.mean(xg_against) if xg_against else 0

        all_team_games = df[
            (((df['HomeTeam']==team)&is_home)|((df['AwayTeam']==team)&(not is_home))) & (df['Date']<date)
        ]
        overall_goals_scored    = all_team_games['FTHG' if is_home else 'FTAG'].mean() if len(all_team_games)>0 else 0
        overall_goals_conceded  = all_team_games['FTAG' if is_home else 'FTHG'].mean() if len(all_team_games)>0 else 0

        return (
            np.nan_to_num(overall_goals_scored),
            np.nan_to_num(overall_goals_conceded),
            np.nan_to_num(recent_goals_scored),
            np.nan_to_num(recent_goals_conceded),
            np.nan_to_num(recent_xg_for),
            np.nan_to_num(recent_xg_against)
        )

    def compute_features(df, reference_df):
        features = [
            'home_goals_scored','home_goals_conceded',
            'away_goals_scored','away_goals_conceded',
            'recent_home_goals_scored','recent_home_goals_conceded',
            'recent_away_goals_scored','recent_away_goals_conceded',
            'home_xg_for','home_xg_against',
            'away_xg_for','away_xg_against',
            'goal_difference',
            'home_form_home','away_form_away',
            'head_to_head_home_win','head_to_head_away_win','head_to_head_draw',
            'HomeTeamPosition','AwayTeamPosition'
        ]
        for feat in features:
            df[feat] = 0.0

        for index,row in df.iterrows():
            (home_goals_scored, home_goals_conceded,
             recent_home_goals_scored, recent_home_goals_conceded,
             home_xg_for, home_xg_against) = team_stats(reference_df, row['HomeTeam'], 'home', row['Date'])
            (away_goals_scored, away_goals_conceded,
             recent_away_goals_scored, recent_away_goals_conceded,
             away_xg_for, away_xg_against) = team_stats(reference_df, row['AwayTeam'], 'away', row['Date'])
            goal_difference = float(home_goals_scored - away_goals_conceded)

            df.at[index,'home_goals_scored']         = home_goals_scored
            df.at[index,'home_goals_conceded']       = home_goals_conceded
            df.at[index,'away_goals_scored']         = away_goals_scored
            df.at[index,'away_goals_conceded']       = away_goals_conceded
            df.at[index,'recent_home_goals_scored']  = recent_home_goals_scored
            df.at[index,'recent_home_goals_conceded']= recent_home_goals_conceded
            df.at[index,'recent_away_goals_scored']  = recent_away_goals_scored
            df.at[index,'recent_away_goals_conceded']= recent_away_goals_conceded
            df.at[index,'home_xg_for']               = home_xg_for
            df.at[index,'home_xg_against']           = home_xg_against
            df.at[index,'away_xg_for']               = away_xg_for
            df.at[index,'away_xg_against']           = away_xg_against
            df.at[index,'goal_difference']           = goal_difference

            h2h_home, h2h_away, h2h_draw = head_to_head(reference_df, row['HomeTeam'], row['AwayTeam'], row['Date'])
            df.at[index,'head_to_head_home_win'] = h2h_home
            df.at[index,'head_to_head_away_win'] = h2h_away
            df.at[index,'head_to_head_draw']     = h2h_draw

        return df

    # Apply feature engineering to historical
    historical_data = compute_features(historical_data, historical_data)
    
    # Also apply to fixtures
    fixtures = compute_features(fixtures, historical_data)

    # Updated feature columns
    feature_columns = [
        'home_goals_scored','home_goals_conceded',
        'away_goals_scored','away_goals_conceded',
        'recent_home_goals_scored','recent_home_goals_conceded',
        'recent_away_goals_scored','recent_away_goals_conceded',
        'home_xg_for','home_xg_against',
        'away_xg_for','away_xg_against',
        'goal_difference',
        'home_form_home','away_form_away',
        'head_to_head_home_win','head_to_head_away_win','head_to_head_draw',
        'HomeTeamPosition','AwayTeamPosition'
    ]

    X = historical_data[feature_columns]
    y = historical_data['FTR_encoded']

    # Split train/test
    split_date = pd.to_datetime('2023-01-01')
    train_data = historical_data[historical_data['Date'] < split_date]
    test_data  = historical_data[historical_data['Date'] >= split_date]

    X_train = train_data[feature_columns]
    y_train = train_data['FTR_encoded']
    X_test  = test_data[feature_columns]
    y_test  = test_data['FTR_encoded']

    # Another reference approach
    reference_data = historical_data[historical_data['Date'] < pd.to_datetime('2024-01-01')]
    fixtures = compute_features(fixtures, reference_data)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Model param grid
    param_grid = {
        'n_estimators': [100,200],
        'max_depth': [None, 20],
        'min_samples_split': [2,10],
        'min_samples_leaf': [1,4]
    }
    class_weights = {0:1.0, 1:1.0, 2:1.0}
    strat_kfold = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight=class_weights),
        param_grid,
        cv=strat_kfold,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    best_rf_model = grid_search.best_estimator_

    # Evaluate
    y_pred = best_rf_model.predict(X_test_scaled)
    # confusion_matrix, classification_report if needed
    # ...
    
    # Fit model on entire data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_rf_model.fit(X_scaled, y)

    # Predict on new fixtures
    fixtures_scaled = scaler.transform(fixtures[feature_columns])
    fixtures['Predicted Result'] = best_rf_model.predict(fixtures_scaled)
    probabilities = best_rf_model.predict_proba(fixtures_scaled)

    fixtures['Prob_H'] = probabilities[:,0].round(4)
    fixtures['Prob_D'] = probabilities[:,1].round(4)
    fixtures['Prob_A'] = probabilities[:,2].round(4)

    reverse_label_mapping = {0:'H', 1:'D', 2:'A'}
    fixtures['Prediction'] = fixtures['Predicted Result'].map(reverse_label_mapping)

    fixtures['High_conf'] = ((fixtures['Prob_H']>0.7)|(fixtures['Prob_A']>0.67)).astype(int)

    fixtures['Weekday'] = fixtures['Date'].dt.day_name()
    # Time adjustments with timezone
    fixtures.loc[:,'Time'] = (
        pd.to_datetime(fixtures['Time'].astype(str), format='%H:%M:%S', errors='coerce')+pd.Timedelta(hours=timezone)
    ).dt.time
    fixtures['Time'] = fixtures['Time'].apply(lambda x: f"{x.hour:02}:{x.minute:02}" if pd.notnull(x) else x)

    # Add double chance columns
    fixtures['double_chance'] = fixtures.apply(
        lambda row: '1X' if row['Prob_H']+row['Prob_D'] > row['Prob_D']+row['Prob_A'] else 'X2',
        axis=1
    )
    fixtures['1X_odds'] = ((fixtures['B365H']*fixtures['B365D'])/(fixtures['B365H']+fixtures['B365D'])).round(2)
    fixtures['X2_odds'] = ((fixtures['B365D']*fixtures['B365A'])/(fixtures['B365D']+fixtures['B365A'])).round(2)
    fixtures['1X_prob'] = (fixtures['Prob_H']+fixtures['Prob_D']).round(4)
    fixtures['X2_prob'] = (fixtures['Prob_D']+fixtures['Prob_A']).round(4)
    fixtures['High_conf_dc'] = ((fixtures['1X_prob']>0.90)|(fixtures['X2_prob']>0.87)).astype(int)

    # Final output
    predictions_df = fixtures[[
        'Div','Date','Weekday','Time','HomeTeam','AwayTeam','Prediction',
        'B365H','B365D','B365A','Prob_H','Prob_D','Prob_A','double_chance',
        '1X_odds','X2_odds','1X_prob','X2_prob','High_conf','High_conf_dc'
    ]].copy()
    predictions_df.sort_values(["Date","Time"], inplace=True)

    # Feature importances (optional)
    feat_importances = best_rf_model.feature_importances_
    # ...
    # (No change if you don't need to print them)

    # Create predictions folder if not exist
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    # Find next available predictions_{N}.csv
    i = 1
    while os.path.exists(os.path.join(PREDICTIONS_DIR, f"predictions_{i}.csv")):
        i += 1

    out_path = os.path.join(PREDICTIONS_DIR, f"predictions_{i}.csv")
    predictions_df.to_csv(out_path, index=False)
    print(f"✅ Predictions exported to {out_path}")

# ------------------------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------------------------
if __name__=="__main__":
    main()
