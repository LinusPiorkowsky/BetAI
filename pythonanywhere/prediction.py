#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Comparison and saving wrong but correct prediction

import os
import re
import io
import requests
import pandas as pd
import zipfile
import numpy as np
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 1) GLOBAL CONFIG (Paths + URLs)
# =============================================================================
BASE_DIR = "/home/MachineLearningBets/BetAI"

SAVE_DIR         = os.path.join(BASE_DIR, "data", "2024_25")  # Where downloaded league CSVs go
FIXTURES_DIR     = os.path.join(BASE_DIR, "data", "fixtures")
PREDICTIONS_DIR  = os.path.join(BASE_DIR, "predictions")
RESULTS_DIR      = os.path.join(BASE_DIR, "results")

ALLOWED_LEAGUES  = {
    "D1.csv","D2.csv","E0.csv","E1.csv","F1.csv","F2.csv",
    "I1.csv","I2.csv","SP1.csv","SP2.csv"
}

DATASET_URL  = "https://www.football-data.co.uk/mmz4281/2425/data.zip"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIXTURES_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Your local definitions
seasons = ['2019_20', '2020_21', '2021_22', '2022_23', '2023_24', '2024_25']
leagues = ['D1', 'F1', 'E0', 'I1', 'SP1']

# Timezone offset
timezone = 1

# =============================================================================
# 2) OPTIONAL: DOWNLOAD FIXTURES
# =============================================================================
def download_fixtures(url: str, filename: str) -> None:
    """
    Download CSV from `url`, filter certain leagues, store in data/fixtures/.
    Columns used: Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,B365>2.5,B365<2.5
    """
    filepath = os.path.join(FIXTURES_DIR, filename)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        if not resp.content:
            print("❌ Fixtures CSV is empty.")
            return

        df = pd.read_csv(io.StringIO(resp.text), encoding="utf-8-sig")
        df.columns = df.columns.str.replace("ï»¿","", regex=True)

        if "Div" not in df.columns:
            raise ValueError(f"❌ 'Div' column missing. Found: {list(df.columns)}")

        # Filter your desired leagues
        allowed = ["D1","E0","F1","I1","SP1"]
        df = df[df["Div"].isin(allowed)]

        keep_cols = ["Div","Date","Time","HomeTeam","AwayTeam","B365H","B365D","B365A","B365>2.5","B365<2.5"]
        df = df[keep_cols]

        df.to_csv(filepath, index=False)
        print(f"✅ Fixtures filtered & saved to {filepath}")

    except requests.RequestException as e:
        print(f"❌ Error downloading fixtures: {e}")
    except Exception as e:
        print(f"❌ Error processing fixtures: {e}")

# =============================================================================
# 3) OPTIONAL: DOWNLOAD DATASET ZIP
# =============================================================================
def download_dataset(url: str, zip_filename: str) -> None:
    """
    Download a ZIP from `url`, extract only ALLOWED_LEAGUES into SAVE_DIR.
    """
    temp_zip = os.path.join(SAVE_DIR, zip_filename)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        if not resp.content:
            print("❌ Dataset is empty.")
            return

        with open(temp_zip, 'wb') as f:
            f.write(resp.content)

        with zipfile.ZipFile(temp_zip, 'r') as zf:
            for member in zf.namelist():
                if any(member.endswith(league) for league in ALLOWED_LEAGUES):
                    zf.extract(member, SAVE_DIR)

        os.remove(temp_zip)
        print(f"✅ Downloaded & extracted -> {SAVE_DIR}")

    except requests.RequestException as e:
        print(f"❌ Error downloading dataset: {e}")
    except zipfile.BadZipFile:
        print("❌ Not a valid ZIP file.")

# =============================================================================
# 4) READ EXISTING PREDICTIONS (skip duplicates)
# =============================================================================
def read_all_predictions(pred_dir: str) -> pd.DataFrame:
    """
    Combines 'predictions_(N).csv' -> single DF with [Date,Time,HomeTeam,AwayTeam]
    so we can skip duplicates if already predicted.
    """
    all_preds = sorted([
        f for f in os.listdir(pred_dir)
        if re.match(r"predictions_(\d+)\.csv$", f)
    ])
    if not all_preds:
        return pd.DataFrame(columns=["Date","Time","HomeTeam","AwayTeam"])

    df_list = []
    for fn in all_preds:
        path = os.path.join(pred_dir, fn)
        pdf = pd.read_csv(path, dayfirst=True, parse_dates=["Date"], encoding="utf-8")
        if "Time" in pdf.columns:
            pdf["Time"] = pdf["Time"].astype(str)
        else:
            pdf["Time"] = ""
        needed = ["Date","Time","HomeTeam","AwayTeam"]
        pdf = pdf[[c for c in needed if c in pdf.columns]]
        df_list.append(pdf)

    combined = pd.concat(df_list, ignore_index=True).drop_duplicates()
    return combined

# =============================================================================
# 5) MAIN
# =============================================================================
def main():
    """
    1) Download fixtures/dataset (optional).
    2) Read historical data (your local logic).
    3) Compare with existing predictions => skip duplicates.
    4) Run your 70% code EXACTLY as is.
    5) Export new predictions_{N}.csv
    """

    # A) Download fixtures/dataset (comment out if not needed)
    download_fixtures(FIXTURES_URL, "fixtures.csv")
    download_dataset(DATASET_URL, "freshdata.zip")

    # -------------------------------------------------------------------------
    # B) Your local logic: Load historical data
    # -------------------------------------------------------------------------
    # EXACT code from your snippet, do NOT change logic:

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.preprocessing import StandardScaler

    # seasons & leagues => you have them above
    # We'll read each data/{season}/{league}.csv
    historical_data_list = []
    for s in seasons:
        for l in leagues:
            path = os.path.join("data", s, f"{l}.csv")
            if os.path.exists(path):
                df_hist = pd.read_csv(path)
                historical_data_list.append(df_hist)
            else:
                print(f"⚠️ {path} not found.")

    if not historical_data_list:
        print("❌ No historical data found. Exiting.")
        return

    historical_data = pd.concat(historical_data_list, ignore_index=True)

    # load fixtures from data/fixtures/fixtures.csv
    fix_path = os.path.join("data","fixtures","fixtures.csv")
    if not os.path.exists(fix_path):
        print(f"❌ No fixtures file found at {fix_path}, aborting.")
        return
    fixtures = pd.read_csv(fix_path)

    # C) Compare with existing predictions => skip duplicates
    existing_preds = read_all_predictions(PREDICTIONS_DIR)
    if not existing_preds.empty:
        # Convert fixtures date format to match predictions (YYYY-MM-DD)
        fixtures['Date'] = pd.to_datetime(fixtures['Date'], dayfirst=True)

        # Standardize time formats in both dataframes
        def fix_time_format(time_str):
            try:
                if pd.isna(time_str) or time_str == '':
                    return ''
                # Handle various time formats
                time_str = str(time_str)
                if ':' in time_str:
                    if len(time_str) <= 5:  # Format like "19:30"
                        return time_str
                    # Format like "19:30:00"
                    return time_str[:5]
                return time_str
            except Exception:
                return ''

        existing_preds['Time'] = existing_preds['Time'].apply(fix_time_format)
        fixtures['Time'] = fixtures['Time'].apply(fix_time_format)

        print(f"✅ Checking {len(fixtures)} fixtures against {len(existing_preds)} existing predictions...")

        # Create matching keys for comparison
        existing_preds['match_key'] = existing_preds.apply(
            lambda x: f"{x['Date'].strftime('%Y-%m-%d')}_{x['Time']}_{x['HomeTeam']}_{x['AwayTeam']}", axis=1)
        fixtures['match_key'] = fixtures.apply(
            lambda x: f"{x['Date'].strftime('%Y-%m-%d')}_{x['Time']}_{x['HomeTeam']}_{x['AwayTeam']}", axis=1)

        # Find duplicates
        duplicate_keys = set(fixtures['match_key']).intersection(set(existing_preds['match_key']))
        duplicates = fixtures['match_key'].isin(duplicate_keys)

        # Filter to new fixtures
        new_fixtures = fixtures[~duplicates].reset_index(drop=True)

        if new_fixtures.empty:
            print("✅ All these fixtures have been predicted. Nothing new.")
            return
        else:
            fixtures = new_fixtures
            print(f"✅ {len(fixtures)} new fixtures to predict.")
    else:
        print("✅ No existing predictions found. All fixtures are new to predict.")

    # -------------------------------------------------------------------------
    # EXACT local snippet (70% accuracy) => do NOT change logic:
    # (We've replaced the inline code with a single block.)
    # -------------------------------------------------------------------------
    # Everything below is your local code:
    from collections import defaultdict

    # Remove duplicates from your snippet if needed
    # define your local logic

    # Timezone offset
    timezone = 1  # same as above, but won't conflict

    # The rest is your exact code:

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

    def calculate_xg(row, is_home=True):
        shots = row['HST' if is_home else 'AST']
        goals = row['FTHG' if is_home else 'FTAG']
        conversion_rate = 0.3  # Average conversion rate for shots on target
        return shots * conversion_rate

    historical_data['xGH'] = historical_data.apply(lambda x: calculate_xg(x, True), axis=1)
    historical_data['xGA'] = historical_data.apply(lambda x: calculate_xg(x, False), axis=1)

    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    historical_data['FTR_encoded'] = historical_data['FTR'].map(label_mapping)

    fixtures['Date'] = pd.to_datetime(fixtures['Date'], dayfirst=True)
    fixtures['DateTime'] = pd.to_datetime(fixtures['Date'].astype(str) + ' ' + fixtures['Time'])
    fixtures['Time'] = fixtures['DateTime'].dt.time
    fixtures.sort_values("Date", inplace=True)

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

            home_pos = sorted(league_table.items(), key=lambda x: (-x[1]['points'], -x[1]['gd'], -x[1]['games']))
            home_rank = next((i + 1 for i, (team, _) in enumerate(home_pos) if team == home_team), len(league_table) + 1)
            away_rank = next((i + 1 for i, (team, _) in enumerate(home_pos) if team == away_team), len(league_table) + 1)

            df.at[idx, 'HomeTeamPosition'] = home_rank
            df.at[idx, 'AwayTeamPosition'] = away_rank

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

            df.at[index, 'home_form_home'] = sum(home_recent) / len(home_recent) if home_recent else 0
            df.at[index, 'away_form_away'] = sum(away_recent) / len(away_recent) if away_recent else 0

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

    def head_to_head(df, home_team, away_team, date, n_games=3):
        past_matches = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
                          ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team)) &
                          (df['Date'] < date)]
        recent_matches = past_matches.tail(n_games)
        home_wins = (recent_matches['HomeTeam'] == home_team) & (recent_matches['FTR'] == 'H')
        away_wins = (recent_matches['AwayTeam'] == home_team) & (recent_matches['FTR'] == 'A')
        draws = (recent_matches['FTR'] == 'D')
        return home_wins.sum(), away_wins.sum(), draws.sum()

    def compute_features(df, reference_df):
        features = [
            'home_goals_scored','home_goals_conceded',
            'away_goals_scored','away_goals_conceded',
            'recent_home_goals_scored','recent_home_goals_conceded',
            'recent_away_goals_scored','recent_away_goals_conceded',
            'home_xg_for','home_xg_against','away_xg_for','away_xg_against',
            'goal_difference','home_form_home','away_form_away',
            'head_to_head_home_win','head_to_head_away_win','head_to_head_draw',
            'HomeTeamPosition','AwayTeamPosition'
        ]

        for feature in features:
            df[feature] = 0.0

        for index, row in df.iterrows():
            (hg_scored, hg_conceded,
             rec_hg_scored, rec_hg_conceded,
             hxg_for, hxg_against) = team_stats(reference_df, row['HomeTeam'], 'home', row['Date'])

            (ag_scored, ag_conceded,
             rec_ag_scored, rec_ag_conceded,
             axg_for, axg_against) = team_stats(reference_df, row['AwayTeam'], 'away', row['Date'])

            goal_difference = float(hg_scored - ag_conceded)

            df.at[index, 'home_goals_scored'] = hg_scored
            df.at[index, 'home_goals_conceded'] = hg_conceded
            df.at[index, 'away_goals_scored'] = ag_scored
            df.at[index, 'away_goals_conceded'] = ag_conceded
            df.at[index, 'recent_home_goals_scored'] = rec_hg_scored
            df.at[index, 'recent_home_goals_conceded'] = rec_hg_conceded
            df.at[index, 'recent_away_goals_scored'] = rec_ag_scored
            df.at[index, 'recent_away_goals_conceded'] = rec_ag_conceded
            df.at[index, 'home_xg_for'] = hxg_for
            df.at[index, 'home_xg_against'] = hxg_against
            df.at[index, 'away_xg_for'] = axg_for
            df.at[index, 'away_xg_against'] = axg_against
            df.at[index, 'goal_difference'] = goal_difference

            h2h_home, h2h_away, h2h_draw = head_to_head(reference_df, row['HomeTeam'], row['AwayTeam'], row['Date'])
            df.at[index, 'head_to_head_home_win'] = h2h_home
            df.at[index, 'head_to_head_away_win'] = h2h_away
            df.at[index, 'head_to_head_draw'] = h2h_draw

        return df

    # Re-apply features:
    historical_data = compute_features(historical_data, historical_data)
    fixtures = compute_features(fixtures, historical_data)

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

    X = historical_data[feature_columns]
    y = historical_data['FTR_encoded']

    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier

    split_date = pd.to_datetime('2023-01-01')
    train_data = historical_data[historical_data['Date'] < split_date]
    test_data = historical_data[historical_data['Date'] >= split_date]

    X_train = train_data[feature_columns]
    y_train = train_data['FTR_encoded']
    X_test = test_data[feature_columns]
    y_test = test_data['FTR_encoded']

    reference_data = historical_data[historical_data['Date'] < pd.Timestamp.today()]
    fixtures = compute_features(fixtures, reference_data)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4]
    }
    class_weights = {0:1.0, 1:1.0, 2:1.0}
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

    y_pred = best_rf_model.predict(X_test_scaled)

    from sklearn.metrics import accuracy_score
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Test Accuracy: {test_accuracy:.3f}")

    # Fit entire data
    X_scaled = scaler.fit_transform(X)
    best_rf_model.fit(X_scaled, y)

    # Predict on fixtures
    fixtures_scaled = scaler.transform(fixtures[feature_columns])
    fixtures['Predicted Result'] = best_rf_model.predict(fixtures_scaled)

    probs = best_rf_model.predict_proba(fixtures_scaled)
    fixtures['Prob_H'] = probs[:, 0].round(4)
    fixtures['Prob_D'] = probs[:, 1].round(4)
    fixtures['Prob_A'] = probs[:, 2].round(4)

    reverse_label_mapping = {0: 'H', 1: 'D', 2: 'A'}
    fixtures['Prediction'] = fixtures['Predicted Result'].map(reverse_label_mapping)

    fixtures['Prob_H'] = fixtures['Prob_H'].round(4)
    fixtures['Prob_D'] = fixtures['Prob_D'].round(4)
    fixtures['Prob_A'] = fixtures['Prob_A'].round(4)

    fixtures['High_conf'] = ((fixtures['Prob_H'] > 0.7) | (fixtures['Prob_A'] > 0.67)).astype(int)
    fixtures['Weekday'] = fixtures['Date'].dt.day_name()

    # Timezones
    fixtures.loc[:, 'Time'] = (
        pd.to_datetime(fixtures['Time'].astype(str), format='%H:%M:%S', errors='coerce')
        + pd.Timedelta(hours=0)
    ).dt.time
    fixtures['Time'] = fixtures['Time'].apply(lambda x: f"{x.hour:02}:{x.minute:02}" if pd.notnull(x) else x)

    fixtures['double_chance'] = fixtures.apply(
        lambda row: '1X' if row['Prob_H'] + row['Prob_D'] > row['Prob_A'] + row['Prob_D'] else 'X2',
        axis=1
    )
    fixtures['1X_odds'] = (
        (fixtures['B365H'] * fixtures['B365D'])
        / (fixtures['B365H'] + fixtures['B365D'])
    ).round(2)
    fixtures['X2_odds'] = (
        (fixtures['B365D'] * fixtures['B365A'])
        / (fixtures['B365D'] + fixtures['B365A'])
    ).round(2)

    fixtures['1X_prob'] = (fixtures['Prob_H'] + fixtures['Prob_D']).round(4)
    fixtures['X2_prob'] = (fixtures['Prob_D'] + fixtures['Prob_A']).round(4)
    fixtures['High_conf_dc'] = ((fixtures['1X_prob'] > 0.90) | (fixtures['X2_prob'] > 0.87)).astype(int)

    predictions_df['Time'] = predictions_df['Time'] + pd.Timedelta(hours=timezone)

    predictions_df = fixtures[[
        'Div', 'Date', 'Weekday', 'Time', 'HomeTeam', 'AwayTeam', 'Prediction',
        'B365H', 'B365D', 'B365A',
        'Prob_H', 'Prob_D', 'Prob_A',
        'double_chance','1X_odds','X2_odds','1X_prob','X2_prob',
        'High_conf','High_conf_dc'
    ]]

    predictions_df['Time'] = predictions_df['Time'] + pd.Timedelta(hours=timezone)
    
    predictions_df.sort_values(["Date","Time"], inplace=True)

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    i = 1
    while os.path.exists(os.path.join(PREDICTIONS_DIR, f"predictions_{i}.csv")):
        i += 1

    out_file = os.path.join(PREDICTIONS_DIR, f"predictions_{i}.csv")
    predictions_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"✅ New predictions saved -> {out_file}")


# =============================================================================
if __name__ == "__main__":
    main()
