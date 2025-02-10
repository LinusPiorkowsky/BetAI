from flask import Flask, render_template, request
import os
import pandas as pd
import datetime
import glob

app = Flask(__name__)

# Define leagues
LEAGUES = ["D1", "F1", "I1", "SP1", "E0"]  # Add more leagues as needed

# Data paths
DATA_DIR = "data"
PREDICTION_DIR = os.path.join(DATA_DIR, "prediction_history")
SEASON_DIR = os.path.join(DATA_DIR, "2024_25")

def get_latest_prediction_file(league):
    """Find the latest prediction file for the given league."""
    pattern = os.path.join(PREDICTION_DIR, f"predictions_{league}_*.csv")
    prediction_files = sorted(glob.glob(pattern), reverse=True)  # Sort newest first
    return prediction_files[0] if prediction_files else None

def load_actual_results(league):
    """Load actual results for the current season."""
    actual_file = os.path.join(SEASON_DIR, f"{league}.csv")
    if os.path.exists(actual_file):
        df_actual = pd.read_csv(actual_file)
        df_actual["Date"] = pd.to_datetime(df_actual["Date"], format="%d/%m/%Y", errors="coerce")
        df_actual = df_actual.dropna(subset=["Date"])  # Remove invalid dates
        return df_actual
    return None

def compare_predictions(league):
    """Compare predictions with actual outcomes."""
    pred_file = get_latest_prediction_file(league)
    actual_data = load_actual_results(league)

    if not pred_file or actual_data is None:
        return None  # No data available

    df_preds = pd.read_csv(pred_file)
    df_preds["Date"] = pd.to_datetime(df_preds["Date"], errors="coerce")
    
    # Filter for last week's matches
    one_week_ago = datetime.datetime.today() - datetime.timedelta(days=7)
    df_preds = df_preds[df_preds["Date"] >= one_week_ago].copy()

    # Merge with actual data
    df_actual = actual_data[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
    df_actual["HomeTeam"] = df_actual["HomeTeam"].str.strip().str.lower()
    df_actual["AwayTeam"] = df_actual["AwayTeam"].str.strip().str.lower()

    df_preds["HomeTeam"] = df_preds["HomeTeam"].str.strip().str.lower()
    df_preds["AwayTeam"] = df_preds["AwayTeam"].str.strip().str.lower()

    df_merged = pd.merge(df_preds, df_actual, on=["Date", "HomeTeam", "AwayTeam"], how="left")

    # Determine if prediction was correct
    df_merged["Correct"] = df_merged["Prediction"] == df_merged["FTR"]

    return df_merged

def load_predictions(league):
    """Load latest predictions only."""
    pred_file = get_latest_prediction_file(league)
    
    if not pred_file:
        return None  # No data available
    
    df_preds = pd.read_csv(pred_file)
    df_preds["Date"] = pd.to_datetime(df_preds["Date"], errors="coerce")
    
    return df_preds

@app.route("/")
def home():
    """Homepage with buttons for each league."""
    return render_template("index.html", leagues=LEAGUES)

@app.route("/league/<league>")
def show_league(league):
    """Display predictions vs actual results for a league."""
    if league not in LEAGUES:
        return "Invalid league!", 404

    results = compare_predictions(league)
    
    if results is None or results.empty:
        return render_template("league.html", league=league, error="No data available.")

    return render_template("league.html", league=league, results=results.to_dict(orient="records"))

@app.route("/predictions/<league>")
def show_predictions(league):
    """Display only the latest predictions for a league."""
    if league not in LEAGUES:
        return "Invalid league!", 404

    predictions = load_predictions(league)

    if predictions is None or predictions.empty:
        return render_template("predictions.html", league=league, error="No predictions available.")

    return render_template("predictions.html", league=league, predictions=predictions.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
