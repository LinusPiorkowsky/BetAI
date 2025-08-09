# prediction.py - Enhanced ML Prediction System
import os
import re
import requests
import pandas as pd
import numpy as np
import zipfile
from datetime import datetime, timedelta
from collections import defaultdict
import logging

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIXTURES_DIR = os.path.join(BASE_DIR, "data", "fixtures")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")
CURRENT_SEASON_DIR = os.path.join(DATA_DIR, "2024_25")

# URLs
DATASET_URL = "https://www.football-data.co.uk/mmz4281/2425/data.zip"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# League configuration
LEAGUES = ['D1', 'F1', 'E0', 'I1', 'SP1']  # Main leagues
SEASONS = ['2019_20', '2020_21', '2021_22', '2022_23', '2023_24', '2024_25']
ALLOWED_LEAGUES = {"D1.csv", "E0.csv", "F1.csv", "I1.csv", "SP1.csv"}

# Create directories
for directory in [DATA_DIR, FIXTURES_DIR, PREDICTIONS_DIR, CURRENT_SEASON_DIR]:
    os.makedirs(directory, exist_ok=True)

class EnhancedFootballPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = [
            'home_goals_scored', 'home_goals_conceded',
            'away_goals_scored', 'away_goals_conceded',
            'recent_home_goals_scored', 'recent_home_goals_conceded',
            'recent_away_goals_scored', 'recent_away_goals_conceded',
            'home_xg_for', 'home_xg_against',
            'away_xg_for', 'away_xg_against',
            'goal_difference', 'form_difference',
            'home_form_home', 'away_form_away',
            'head_to_head_home_win', 'head_to_head_away_win', 'head_to_head_draw',
            'home_team_position', 'away_team_position', 'position_difference',
            'home_shots_ratio', 'away_shots_ratio',
            'home_defense_strength', 'away_attack_strength'
        ]

    def download_data(self):
        """Download latest fixtures and results."""
        try:
            # Download fixtures
            logger.info("Downloading fixtures...")
            resp = requests.get(FIXTURES_URL, timeout=30)
            resp.raise_for_status()
            
            fixtures_df = pd.read_csv(resp.text.splitlines())
            fixtures_df = fixtures_df[fixtures_df['Div'].isin(LEAGUES)]
            
            keep_cols = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", 
                        "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"]
            fixtures_df = fixtures_df[[col for col in keep_cols if col in fixtures_df.columns]]
            
            fixtures_path = os.path.join(FIXTURES_DIR, "fixtures.csv")
            fixtures_df.to_csv(fixtures_path, index=False)
            logger.info(f"Fixtures saved to {fixtures_path}")

            # Download current season data
            logger.info("Downloading current season data...")
            resp = requests.get(DATASET_URL, timeout=30)
            resp.raise_for_status()
            
            zip_path = os.path.join(CURRENT_SEASON_DIR, "data.zip")
            with open(zip_path, 'wb') as f:
                f.write(resp.content)
            
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in zf.namelist():
                    if any(member.endswith(league) for league in ALLOWED_LEAGUES):
                        zf.extract(member, CURRENT_SEASON_DIR)
            
            os.remove(zip_path)
            logger.info("Current season data downloaded successfully")

        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise

    def load_historical_data(self):
        """Load and combine historical data from multiple seasons."""
        historical_data_list = []
        
        for season in SEASONS:
            for league in LEAGUES:
                path = os.path.join(DATA_DIR, season, f"{league}.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df['Season'] = season
                    historical_data_list.append(df)
                else:
                    logger.warning(f"File not found: {path}")
        
        if not historical_data_list:
            raise ValueError("No historical data found")
        
        historical_data = pd.concat(historical_data_list, ignore_index=True)
        
        # Clean and prepare data
        historical_data = historical_data[historical_data['FTR'].isin(['H', 'D', 'A'])]
        historical_data['Date'] = pd.to_datetime(historical_data['Date'], dayfirst=True)
        
        required_columns = [
            'Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
            'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'B365H', 'B365D', 'B365A'
        ]
        
        # Keep only columns that exist
        available_columns = [col for col in required_columns if col in historical_data.columns]
        historical_data = historical_data[available_columns].dropna()
        
        historical_data = historical_data.sort_values("Date").reset_index(drop=True)
        logger.info(f"Loaded {len(historical_data)} historical matches")
        
        return historical_data

    def calculate_enhanced_xg(self, df):
        """Calculate enhanced expected goals using shots and other metrics."""
        def xg_calculation(row, is_home=True):
            shots = row.get('HST' if is_home else 'AST', 0)
            total_shots = row.get('HS' if is_home else 'AS', shots)
            corners = row.get('HC' if is_home else 'AC', 0)
            
            # Base conversion rate
            conversion_rate = 0.35 if shots > 0 else 0.1
            
            # Adjust for shot accuracy
            if total_shots > 0:
                accuracy = shots / total_shots
                conversion_rate *= (0.5 + accuracy)
            
            # Corner bonus
            corner_bonus = corners * 0.1
            
            return max(0.1, shots * conversion_rate + corner_bonus)
        
        df['xGH'] = df.apply(lambda x: xg_calculation(x, True), axis=1)
        df['xGA'] = df.apply(lambda x: xg_calculation(x, False), axis=1)
        
        return df

    def calculate_team_positions(self, df):
        """Calculate league positions throughout the season."""
        positions = {}
        
        for div in df['Div'].unique():
            div_data = df[df['Div'] == div].sort_values('Date')
            
            for season in div_data['Season'].unique() if 'Season' in div_data.columns else [None]:
                season_data = div_data[div_data['Season'] == season] if season else div_data
                league_table = defaultdict(lambda: {'points': 0, 'games': 0, 'gd': 0})
                
                for idx, row in season_data.iterrows():
                    home_team, away_team = row['HomeTeam'], row['AwayTeam']
                    
                    # Get current positions
                    sorted_teams = sorted(league_table.items(), 
                                        key=lambda x: (-x[1]['points'], -x[1]['gd']))
                    
                    home_pos = next((i + 1 for i, (team, _) in enumerate(sorted_teams) 
                                   if team == home_team), len(league_table) + 1)
                    away_pos = next((i + 1 for i, (team, _) in enumerate(sorted_teams) 
                                   if team == away_team), len(league_table) + 1)
                    
                    df.at[idx, 'home_team_position'] = home_pos
                    df.at[idx, 'away_team_position'] = away_pos
                    
                    # Update table
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

    def calculate_team_stats(self, df, team, is_home, date, n_matches=5):
        """Calculate comprehensive team statistics."""
        # Filter team games before the given date
        if is_home:
            team_games = df[(df['HomeTeam'] == team) & (df['Date'] < date)]
        else:
            team_games = df[(df['AwayTeam'] == team) & (df['Date'] < date)]
        
        if len(team_games) == 0:
            return [0] * 8  # Return zeros for all stats
        
        # Recent form (last n_matches)
        recent_games = team_games.tail(n_matches)
        
        # Calculate stats
        if is_home:
            goals_scored = team_games['FTHG'].mean()
            goals_conceded = team_games['FTAG'].mean()
            recent_goals_scored = recent_games['FTHG'].mean() if len(recent_games) > 0 else 0
            recent_goals_conceded = recent_games['FTAG'].mean() if len(recent_games) > 0 else 0
            xg_for = team_games['xGH'].mean()
            xg_against = team_games['xGA'].mean()
            shots_ratio = (team_games['HST'].sum() / team_games['HS'].sum()) if team_games['HS'].sum() > 0 else 0.3
        else:
            goals_scored = team_games['FTAG'].mean()
            goals_conceded = team_games['FTHG'].mean()
            recent_goals_scored = recent_games['FTAG'].mean() if len(recent_games) > 0 else 0
            recent_goals_conceded = recent_games['FTHG'].mean() if len(recent_games) > 0 else 0
            xg_for = team_games['xGA'].mean()
            xg_against = team_games['xGH'].mean()
            shots_ratio = (team_games['AST'].sum() / team_games['AS'].sum()) if team_games['AS'].sum() > 0 else 0.3
        
        # Defense and attack strength
        defense_strength = max(0.1, 2.0 - goals_conceded)
        attack_strength = min(3.0, goals_scored)
        
        return [
            goals_scored, goals_conceded, recent_goals_scored, recent_goals_conceded,
            xg_for, xg_against, shots_ratio, defense_strength if is_home else attack_strength
        ]

    def calculate_form(self, df, n_games=5):
        """Calculate team form over recent games."""
        team_form = defaultdict(list)
        
        for idx, row in df.iterrows():
            home_team, away_team = row['HomeTeam'], row['AwayTeam']
            
            # Get recent form
            home_form = team_form[home_team][-n_games:] if team_form[home_team] else []
            away_form = team_form[away_team][-n_games:] if team_form[away_team] else []
            
            df.at[idx, 'home_form_home'] = sum(home_form) / len(home_form) if home_form else 0.5
            df.at[idx, 'away_form_away'] = sum(away_form) / len(away_form) if away_form else 0.5
            
            # Update form based on result
            if row['FTR'] == 'H':
                team_form[home_team].append(1)
                team_form[away_team].append(0)
            elif row['FTR'] == 'A':
                team_form[home_team].append(0)
                team_form[away_team].append(1)
            else:
                team_form[home_team].append(0.5)
                team_form[away_team].append(0.5)
        
        return df

    def calculate_head_to_head(self, df, home_team, away_team, date, n_games=5):
        """Calculate head-to-head record."""
        h2h_matches = df[
            (((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
             ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))) &
            (df['Date'] < date)
        ].tail(n_games)
        
        if len(h2h_matches) == 0:
            return 0, 0, 0
        
        home_wins = len(h2h_matches[
            ((h2h_matches['HomeTeam'] == home_team) & (h2h_matches['FTR'] == 'H')) |
            ((h2h_matches['AwayTeam'] == home_team) & (h2h_matches['FTR'] == 'A'))
        ])
        
        away_wins = len(h2h_matches[
            ((h2h_matches['HomeTeam'] == away_team) & (h2h_matches['FTR'] == 'H')) |
            ((h2h_matches['AwayTeam'] == away_team) & (h2h_matches['FTR'] == 'A'))
        ])
        
        draws = len(h2h_matches[h2h_matches['FTR'] == 'D'])
        
        return home_wins, away_wins, draws

    def compute_features(self, df, reference_df):
        """Compute all features for the dataset."""
        logger.info("Computing features...")
        
        # Initialize feature columns
        for feature in self.feature_columns:
            df[feature] = 0.0
        
        for idx, row in df.iterrows():
            date = row['Date']
            home_team, away_team = row['HomeTeam'], row['AwayTeam']
            
            # Team statistics
            home_stats = self.calculate_team_stats(reference_df, home_team, True, date)
            away_stats = self.calculate_team_stats(reference_df, away_team, False, date)
            
            # Assign basic stats
            (df.at[idx, 'home_goals_scored'], df.at[idx, 'home_goals_conceded'],
             df.at[idx, 'recent_home_goals_scored'], df.at[idx, 'recent_home_goals_conceded'],
             df.at[idx, 'home_xg_for'], df.at[idx, 'home_xg_against'],
             df.at[idx, 'home_shots_ratio'], df.at[idx, 'home_defense_strength']) = home_stats
            
            (df.at[idx, 'away_goals_scored'], df.at[idx, 'away_goals_conceded'],
             df.at[idx, 'recent_away_goals_scored'], df.at[idx, 'recent_away_goals_conceded'],
             df.at[idx, 'away_xg_for'], df.at[idx, 'away_xg_against'],
             df.at[idx, 'away_shots_ratio'], df.at[idx, 'away_attack_strength']) = away_stats
            
            # Derived features
            df.at[idx, 'goal_difference'] = home_stats[0] - away_stats[1]
            df.at[idx, 'form_difference'] = row.get('home_form_home', 0.5) - row.get('away_form_away', 0.5)
            df.at[idx, 'position_difference'] = row.get('away_team_position', 10) - row.get('home_team_position', 10)
            
            # Head-to-head
            h2h_home, h2h_away, h2h_draw = self.calculate_head_to_head(
                reference_df, home_team, away_team, date
            )
            df.at[idx, 'head_to_head_home_win'] = h2h_home
            df.at[idx, 'head_to_head_away_win'] = h2h_away
            df.at[idx, 'head_to_head_draw'] = h2h_draw
        
        return df

    def create_ensemble_model(self):
        """Create an ensemble model combining multiple algorithms."""
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        lr = LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='ovr'
        )
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
            voting='soft'
        )
        
        return ensemble

    def train_model(self, X_train, y_train):
        """Train the ensemble model with hyperparameter tuning."""
        logger.info("Training ensemble model...")
        
        # Create and train ensemble
        self.model = self.create_ensemble_model()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        logger.info("Model training completed")

    def predict_matches(self, fixtures_df):
        """Make predictions for upcoming matches."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        logger.info(f"Making predictions for {len(fixtures_df)} matches...")
        
        # Scale features
        X_fixtures = self.scaler.transform(fixtures_df[self.feature_columns])
        
        # Make predictions
        predictions = self.model.predict(X_fixtures)
        probabilities = self.model.predict_proba(X_fixtures)
        
        # Add predictions to dataframe
        label_mapping = {0: 'H', 1: 'D', 2: 'A'}
        fixtures_df['Prediction'] = [label_mapping[p] for p in predictions]
        fixtures_df['Prob_H'] = probabilities[:, 0]
        fixtures_df['Prob_D'] = probabilities[:, 1]
        fixtures_df['Prob_A'] = probabilities[:, 2]
        
        # Calculate confidence levels
        max_probs = np.max(probabilities, axis=1)
        fixtures_df['Confidence'] = max_probs
        
        # High confidence criteria (more conservative)
        fixtures_df['High_conf'] = (
            ((fixtures_df['Prob_H'] > 0.65) & (fixtures_df['B365H'] < 2.0)) |
            ((fixtures_df['Prob_A'] > 0.65) & (fixtures_df['B365A'] < 2.0)) |
            ((fixtures_df['Prob_D'] > 0.45) & (fixtures_df['B365D'] < 3.5))
        ).astype(int)
        
        return fixtures_df

    def calculate_betting_odds(self, fixtures_df):
        """Calculate various betting options and their odds."""
        # Double chance calculations
        fixtures_df['1X_prob'] = fixtures_df['Prob_H'] + fixtures_df['Prob_D']
        fixtures_df['X2_prob'] = fixtures_df['Prob_D'] + fixtures_df['Prob_A']
        
        # Calculate implied odds for double chance
        fixtures_df['1X_odds'] = np.where(
            (fixtures_df['B365H'].notna()) & (fixtures_df['B365D'].notna()),
            (fixtures_df['B365H'] * fixtures_df['B365D']) / 
            (fixtures_df['B365H'] + fixtures_df['B365D']),
            2.0
        )
        
        fixtures_df['X2_odds'] = np.where(
            (fixtures_df['B365D'].notna()) & (fixtures_df['B365A'].notna()),
            (fixtures_df['B365D'] * fixtures_df['B365A']) / 
            (fixtures_df['B365D'] + fixtures_df['B365A']),
            2.0
        )
        
        # Determine best double chance bet
        fixtures_df['double_chance'] = np.where(
            fixtures_df['1X_prob'] > fixtures_df['X2_prob'], '1X', 'X2'
        )
        
        # High confidence double chance
        fixtures_df['High_conf_dc'] = (
            (fixtures_df['1X_prob'] > 0.85) | (fixtures_df['X2_prob'] > 0.85)
        ).astype(int)
        
        return fixtures_df

def load_fixtures():
    """Load upcoming fixtures."""
    fixtures_path = os.path.join(FIXTURES_DIR, "fixtures.csv")
    if not os.path.exists(fixtures_path):
        raise FileNotFoundError(f"Fixtures file not found: {fixtures_path}")
    
    fixtures = pd.read_csv(fixtures_path)
    fixtures['Date'] = pd.to_datetime(fixtures['Date'], dayfirst=True)
    fixtures = fixtures.sort_values('Date')
    
    # Filter future matches only
    today = datetime.now()
    fixtures = fixtures[fixtures['Date'] >= today.date()]
    
    logger.info(f"Loaded {len(fixtures)} upcoming fixtures")
    return fixtures

def read_existing_predictions():
    """Read existing predictions to avoid duplicates."""
    prediction_files = sorted([
        f for f in os.listdir(PREDICTIONS_DIR)
        if f.startswith("predictions_") and f.endswith(".csv")
    ])
    
    if not prediction_files:
        return pd.DataFrame(columns=["Date", "Time", "HomeTeam", "AwayTeam"])
    
    df_list = []
    for file in prediction_files:
        path = os.path.join(PREDICTIONS_DIR, file)
        df = pd.read_csv(path, parse_dates=["Date"])
        needed_cols = ["Date", "Time", "HomeTeam", "AwayTeam"]
        df = df[[col for col in needed_cols if col in df.columns]]
        df_list.append(df)
    
    return pd.concat(df_list, ignore_index=True).drop_duplicates()

def get_next_prediction_number():
    """Get the next prediction file number."""
    existing = [
        f for f in os.listdir(PREDICTIONS_DIR)
        if f.startswith("predictions_") and f.endswith(".csv")
    ]
    
    if not existing:
        return 1
    
    numbers = []
    for file in existing:
        match = re.search(r"predictions_(\d+)\.csv", file)
        if match:
            numbers.append(int(match.group(1)))
    
    return max(numbers) + 1 if numbers else 1

def main():
    """Main prediction pipeline."""
    try:
        logger.info("Starting prediction pipeline...")
        
        # Initialize predictor
        predictor = EnhancedFootballPredictor()
        
        # Download latest data
        predictor.download_data()
        
        # Load and prepare historical data
        historical_data = predictor.load_historical_data()
        historical_data = predictor.calculate_enhanced_xg(historical_data)
        historical_data = predictor.calculate_team_positions(historical_data)
        historical_data = predictor.calculate_form(historical_data)
        
        # Compute features for historical data
        historical_data = predictor.compute_features(historical_data, historical_data)
        
        # Prepare training data
        X = historical_data[predictor.feature_columns]
        y = historical_data['FTR'].map({'H': 0, 'D': 1, 'A': 2})
        
        # Split data chronologically
        split_date = pd.to_datetime('2023-01-01')
        train_mask = historical_data['Date'] < split_date
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[~train_mask], y[~train_mask]
        
        # Train model
        predictor.train_model(X_train, y_train)
        
        # Evaluate on test set
        X_test_scaled = predictor.scaler.transform(X_test)
        test_predictions = predictor.model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, test_predictions)
        logger.info(f"Test accuracy: {test_accuracy:.3f}")
        
        # Load fixtures
        fixtures = load_fixtures()
        
        # Check for existing predictions
        existing_predictions = read_existing_predictions()
        if not existing_predictions.empty:
            # Remove duplicates
            fixtures_key = fixtures.apply(
                lambda x: f"{x['Date'].strftime('%Y-%m-%d')}_{x.get('Time', '')}_{x['HomeTeam']}_{x['AwayTeam']}", 
                axis=1
            )
            existing_key = existing_predictions.apply(
                lambda x: f"{x['Date'].strftime('%Y-%m-%d')}_{x.get('Time', '')}_{x['HomeTeam']}_{x['AwayTeam']}", 
                axis=1
            )
            
            new_fixtures = fixtures[~fixtures_key.isin(existing_key)]
            logger.info(f"Found {len(new_fixtures)} new fixtures to predict")
        else:
            new_fixtures = fixtures
        
        if new_fixtures.empty:
            logger.info("No new fixtures to predict")
            return
        
        # Compute features for fixtures
        new_fixtures = predictor.compute_features(new_fixtures, historical_data)
        
        # Make predictions
        new_fixtures = predictor.predict_matches(new_fixtures)
        new_fixtures = predictor.calculate_betting_odds(new_fixtures)
        
        # Add metadata
        new_fixtures['Weekday'] = new_fixtures['Date'].dt.day_name()
        
        # Adjust time for timezone (add 1 hour)
        if 'Time' in new_fixtures.columns:
            new_fixtures['Time'] = pd.to_datetime(
                new_fixtures['Time'], format='%H:%M', errors='coerce'
            ) + timedelta(hours=1)
            new_fixtures['Time'] = new_fixtures['Time'].dt.strftime('%H:%M')
        
        # Select final columns
        final_columns = [
            'Div', 'Date', 'Weekday', 'Time', 'HomeTeam', 'AwayTeam', 'Prediction',
            'B365H', 'B365D', 'B365A', 'Prob_H', 'Prob_D', 'Prob_A',
            'double_chance', '1X_odds', 'X2_odds', '1X_prob', 'X2_prob',
            'High_conf', 'High_conf_dc', 'Confidence'
        ]
        
        predictions_df = new_fixtures[[col for col in final_columns if col in new_fixtures.columns]]
        predictions_df = predictions_df.sort_values(['Date', 'Time'])
        
        # Save predictions
        next_num = get_next_prediction_number()
        output_file = os.path.join(PREDICTIONS_DIR, f"predictions_{next_num}.csv")
        predictions_df.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(predictions_df)} predictions to {output_file}")
        
        # Print summary
        high_conf_count = predictions_df['High_conf'].sum()
        logger.info(f"High confidence predictions: {high_conf_count}/{len(predictions_df)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {e}")
        raise

if __name__ == "__main__":
    main()